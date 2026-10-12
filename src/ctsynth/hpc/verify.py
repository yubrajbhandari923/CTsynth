"""Validate a bundle or result directory. Safe on untrusted input: pure reads of JSON/YAML and
file bytes plus `git rev-parse`; never executes, imports, or shells out to anything in the dir."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VerifyReport:
    kind: str  # "bundle" | "result" | "unknown"
    problems: list[str] = field(default_factory=list)
    bundle: dict[str, Any] = field(default_factory=dict)  # parsed bundle.json ({} if unreadable)
    result: dict[str, Any] = field(default_factory=dict)  # parsed result.json ({} for bundles)

    @property
    def ok(self) -> bool:
        return not self.problems


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
    raise NotImplementedError
