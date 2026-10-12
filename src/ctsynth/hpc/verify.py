"""Validate a bundle or result directory. Safe on untrusted input: pure reads of JSON/YAML and
file bytes plus `git rev-parse`; never executes, imports, or shells out to anything in the dir."""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ctsynth.config import ExperimentConfig
from ctsynth.hpc.bundle import (
    BUNDLE_KEYS,
    BUNDLE_MANIFEST,
    RESULT_KEYS,
    RESULT_MANIFEST,
    SCHEMA_VERSION,
    check_manifest,
    sha256_file,
)


@dataclass
class VerifyReport:
    kind: str  # "bundle" | "result" | "unknown"
    problems: list[str] = field(default_factory=list)
    bundle: dict[str, Any] = field(default_factory=dict)  # parsed bundle.json ({} if unreadable)
    result: dict[str, Any] = field(default_factory=dict)  # parsed result.json ({} for bundles)

    @property
    def ok(self) -> bool:
        return not self.problems


def _load_json(path: Path, problems: list[str]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        problems.append(f"{path.name} is unreadable or not valid JSON")
        return None
    if not isinstance(data, dict):
        problems.append(f"{path.name} is not a JSON object")
        return None
    return data


def _check_file(
    base: Path, rel: str, nbytes: object, digest: object, label: str, problems: list[str]
) -> None:
    f = base / rel
    if not f.is_file() or f.is_symlink():
        problems.append(f"{label} {rel} missing")
        return
    if f.stat().st_size != nbytes or sha256_file(f) != digest:
        problems.append(f"{label} {rel}: bytes/sha256 mismatch against result.json")


def _check_commit(sha: object, repo: Path | None, problems: list[str]) -> None:
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{4,39}", sha):
            problems.append(f"git_commit {sha!r} is ambiguous (abbreviated; need the full sha)")
        else:
            problems.append(f"git_commit {sha!r} is not a valid 40-hex sha (unknown commit)")
        return
    if repo is None:
        return  # worker has no repo: existence is checked by the laptop import instead
    # sha is pre-validated as 40 hex, so the argv carries no injection surface
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
        cwd=repo, capture_output=True, text=True, check=False,
    )  # fmt: skip
    if proc.returncode != 0 or proc.stdout.strip() != sha:
        problems.append(f"git_commit {sha} is an unknown commit in {repo}")


def _check_config(path: Path, bundle: dict[str, Any], problems: list[str]) -> None:
    cfg_file = path / "config.yaml"
    try:
        data = yaml.safe_load(cfg_file.read_bytes())
        cfg = ExperimentConfig.model_validate(data)
    except (OSError, yaml.YAMLError, ValueError, TypeError):
        problems.append("config.yaml is unreadable or does not validate as ExperimentConfig")
        return
    expected = bundle.get("config_sha256")
    actual = _sha_of_config(cfg)
    if actual != expected:
        problems.append(
            f"config.yaml: config_sha256 mismatch (bundle records {str(expected)[:12]}, "
            f"file gives {actual[:12]})"
        )
    try:
        embedded = ExperimentConfig.model_validate(bundle.get("config"))
    except (ValueError, TypeError):
        problems.append("embedded config in bundle.json does not validate")
        return
    if _sha_of_config(embedded) != expected:
        problems.append("embedded config in bundle.json does not match config_sha256")


def _sha_of_config(cfg: ExperimentConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()


def _check_bundle(
    path: Path,
    bundle: dict[str, Any],
    report: VerifyReport,
    repo: Path | None,
    data_root: Path | None,
) -> None:
    problems = report.problems
    missing = [k for k in BUNDLE_KEYS if k not in bundle]
    if missing:
        problems.append(f"bundle.json missing keys: {', '.join(missing)}")
        return
    if bundle["schema_version"] != SCHEMA_VERSION:
        problems.append(f"unsupported schema_version {bundle['schema_version']!r}")
    if bundle["git_dirty_source"] is not False:
        problems.append("git_dirty_source is not false: the source was dirty, not citable")
    _check_commit(bundle["git_commit"], repo, problems)
    _check_config(path, bundle, problems)
    if data_root is not None:
        for entry in bundle["data"]:
            name = str(entry.get("path", ""))
            if name in ("", ".", "..") or "/" in name:
                problems.append(f"data entry has unsafe name {name!r}")
                continue
            f = data_root / name
            if not f.is_file():
                problems.append(f"data {name} missing under data_root")
            elif f.stat().st_size != entry.get("bytes") or sha256_file(f) != entry.get("sha256"):
                problems.append(f"data {name}: bytes/sha256 mismatch against bundle")


def _check_result(path: Path, result: dict[str, Any], report: VerifyReport) -> None:
    problems = report.problems
    bundle = report.bundle
    missing = [k for k in RESULT_KEYS if k not in result]
    if missing:
        problems.append(f"result.json missing keys: {', '.join(missing)}")
        return
    for key in ("bundle_id", "experiment_id", "git_commit", "config_sha256"):
        if result[key] != bundle.get(key):
            problems.append(f"result.json {key} disagrees with bundle.json")
    status, code = result["status"], result["exit_code"]
    if status not in ("completed", "failed"):
        problems.append(f"unknown status {status!r}")
    elif status != ("completed" if code == 0 else "failed"):
        problems.append(f"status {status!r} inconsistent with exit_code {code}")
    for out in result["outputs"]:
        _check_file(path, out["path"], out["bytes"], out["sha256"], "output", problems)
    for ck in result["checkpoints"]:
        if not ck["external"]:  # external files live elsewhere; only their record is checkable
            _check_file(path, ck["path"], ck["bytes"], ck["sha256"], "checkpoint", problems)
    if status != "completed":
        return  # a failed run needs none of its outputs
    produced = {ck["path"] for ck in result["checkpoints"]}
    for req in bundle.get("required_outputs", []):
        if not (path / req).is_file():
            problems.append(f"required output {req} missing")
    for ck in bundle.get("checkpoints", []):
        if ck["path"] not in produced:
            problems.append(f"declared checkpoint {ck['path']} was not produced (missing)")


def verify(path: Path, *, repo: Path | None = None, data_root: Path | None = None) -> VerifyReport:
    """Collect ALL problems (never raise on bad content), each a precise human-readable string.

    kind = "result" if result.json exists else "bundle" if bundle.json exists else "unknown".
    Checks: manifest (BUNDLE_MANIFEST or RESULT_MANIFEST) via check_manifest; bundle.json
    parses and has schema_version == SCHEMA_VERSION and every BUNDLE_KEYS key; config.yaml
    loads and its ExperimentConfig JSON sha256 == config_sha256 == the embedded `config`'s;
    git_dirty_source is false; git_commit is 40 lowercase hex (abbreviations are "ambiguous")
    and, when `repo` is given, `git rev-parse --verify <sha>^{commit}` returns exactly that sha
    ("unknown commit" otherwise); with repo=None the existence check is skipped, not passed.
    data_root: each bundle data entry must exist under it with matching bytes+sha256.
    Result extras: result.json has RESULT_KEYS and agrees with bundle.json on bundle_id,
    experiment_id, git_commit, config_sha256; status/exit_code consistent (completed <=> 0);
    every listed output and non-external checkpoint exists with matching hash; when status is
    "completed" every bundle required_outputs and every declared checkpoint is present;
    a "failed" result does NOT need its required outputs.
    """
    path = Path(path)
    report = VerifyReport(kind="unknown")
    if not path.is_dir():
        report.problems.append(f"{path} is not a directory")
        return report
    if (path / "result.json").is_file():
        report.kind = "result"
    elif (path / "bundle.json").is_file():
        report.kind = "bundle"
    else:
        report.problems.append("neither bundle.json nor result.json found")
        return report
    manifest = RESULT_MANIFEST if report.kind == "result" else BUNDLE_MANIFEST
    report.problems += check_manifest(path, manifest)
    bundle = _load_json(path / "bundle.json", report.problems)
    if bundle is None:
        return report
    report.bundle = bundle
    _check_bundle(path, bundle, report, repo, data_root)
    if report.kind == "result":
        result = _load_json(path / "result.json", report.problems)
        if result is None:
            return report
        report.result = result
        _check_result(path, result, report)
    return report
