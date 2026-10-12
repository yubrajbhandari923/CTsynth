"""Bundle creation (laptop), SLURM rendering, result export (worker). See docs/hpc-bundle-spec.md.

Nothing here touches the network. `export_result` runs on the worker and must not need git,
Research OS, or write anywhere except its `out_dir`.
"""

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from ctsynth.config import ExperimentConfig, load_config

SCHEMA_VERSION = 1
UNSET = "unset"  # documented default of every SLURM/hardware field
BUNDLE_MANIFEST = "MANIFEST.sha256"  # covers every file in a bundle dir except itself
RESULT_MANIFEST = "RESULT.sha256"  # same, for a result dir
HARDWARE_FIELDS = (
    "partition", "account", "gpus", "cpus", "mem", "time", "job_name", "scratch",
)  # fmt: skip
# bundle.json top-level keys (see spec for types)
BUNDLE_KEYS = (
    "schema_version", "bundle_id", "experiment_id", "created_utc", "git_commit",
    "git_dirty_source", "config", "config_sha256", "seeds", "data", "uv_lock_sha256",
    "hardware", "env_setup", "entry_point", "required_outputs", "checkpoints",
)  # fmt: skip
# result.json top-level keys
RESULT_KEYS = (
    "schema_version", "bundle_id", "experiment_id", "git_commit", "config_sha256", "status",
    "exit_code", "started_utc", "ended_utc", "slurm_job_id", "outputs", "checkpoints",
)  # fmt: skip
# Files export_result copies in from the bundle; they are not job outputs.
_BUNDLE_COPIES = ("bundle.json", "config.yaml")
_MANIFEST_LINE = re.compile(r"^([0-9a-f]{64}) {2}(.+)$")


class BundleError(Exception):
    """prepare/export refused. The message names the precise reason."""


def sha256_file(path: Path) -> str:
    # Same pattern as projects/p02_aapm_benchmark/evaluate.py (src/ cannot import projects/).
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _regular_files(directory: Path) -> tuple[set[str], list[str]]:
    """POSIX relpaths of regular files under `directory`, and of symlinks (never followed)."""
    files: set[str] = set()
    links: list[str] = []
    for dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            rel = p.relative_to(directory).as_posix()
            if p.is_symlink():
                links.append(rel)
            elif name in filenames:
                files.add(rel)
    return files, links


def write_manifest(directory: Path, name: str) -> Path:
    """Write `<directory>/<name>`: sorted `<sha256>  <posix relpath>` lines (sha256sum format)
    for every regular file under `directory` except the manifest itself."""
    files, _ = _regular_files(directory)
    lines = [f"{sha256_file(directory / rel)}  {rel}" for rel in sorted(files - {name})]
    out = directory / name
    out.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return out


def check_manifest(directory: Path, name: str) -> list[str]:
    """Return problems (empty = ok): manifest missing/malformed, listed file missing, hash
    mismatch, file present but unlisted, absolute/`..`/symlink paths. Pure read; never raises
    on bad content."""
    mpath = directory / name
    if mpath.is_symlink() or not mpath.is_file():
        return [f"{name} missing"]
    try:
        text = mpath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return [f"{name} unreadable"]
    problems: list[str] = []
    listed: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        m = _MANIFEST_LINE.match(line)
        if not m:
            problems.append(f"{name} line {lineno} malformed")
            continue
        digest, rel = m.groups()
        p = PurePosixPath(rel)
        if p.is_absolute() or ".." in p.parts or rel in ("", "."):
            problems.append(f"{name}: unsafe path {rel!r}")
        elif rel in listed:
            problems.append(f"{name}: duplicate entry {rel}")
        else:
            listed[rel] = digest
    present, links = _regular_files(directory)
    problems += [f"symlink not allowed: {rel}" for rel in links]
    for rel, digest in sorted(listed.items()):
        if rel not in present:
            problems.append(f"{rel}: listed in {name} but missing")
            continue
        try:
            actual = sha256_file(directory / rel)
        except OSError:
            problems.append(f"{rel}: unreadable")
            continue
        if actual != digest:
            problems.append(
                f"{rel}: sha256 mismatch (manifest {digest[:12]}, actual {actual[:12]})"
            )
    for rel in sorted(present - set(listed) - {name}):
        problems.append(f"{rel}: not listed in {name} (unlisted file)")
    return problems


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # argv list, no shell; cwd is the repo, never the bundle/result directory
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)


def _config_sha256(cfg: ExperimentConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()


def prepare(
    repo: Path,
    config_path: Path,
    out_root: Path,
    *,
    entry_point: Sequence[str],
    required_outputs: Sequence[str],
    data_files: Sequence[Path] = (),
    checkpoints: Sequence[Mapping[str, Any]] = (),
    hardware: Mapping[str, str] | None = None,
    env_setup: Sequence[str] = (),
    now: datetime | None = None,
) -> Path:
    """Create `<out_root>/<EXP>-<sha8>-<YYYYmmddTHHMMSSZ>/` and return it.

    Contents: bundle.json, config.yaml (byte copy of config_path), run.sbatch, MANIFEST.sha256.
    entry_point is argv with `{config}` / `{out}` placeholders (see resolve_entry); its first
    element is a bare command such as "python"; the path is relative to the source root.
    checkpoints: [{"path": str, "external": bool}] declared in advance (sizes/hashes are
    unknown until the run; they are measured by export_result).
    hardware: any subset of HARDWARE_FIELDS; missing ones become UNSET.
    Raises BundleError (and creates NOTHING on disk) if: git_dirty_source is true (same
    pathspec as runmeta: src tests configs, untracked files included); repo has no uv.lock;
    config fails ctsynth.config.load_config; experiment_id empty; the target directory already
    exists (never overwrite; `now` is injectable for tests); a data file is missing.
    Writes the bundle into a sibling temp dir and renames it into place (atomic).
    """
    repo = Path(repo)
    dirty = _git(repo, "status", "--porcelain", "--", "src", "tests", "configs")
    if dirty.returncode != 0:
        raise BundleError(f"not a git repository: {repo}")
    if dirty.stdout.strip():
        raise BundleError(
            "git_dirty_source: uncommitted changes under src/tests/configs; commit first:\n"
            + dirty.stdout.rstrip()
        )
    head = _git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    commit = head.stdout.strip()
    if head.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise BundleError(f"cannot resolve HEAD of {repo} to a full commit")
    lock = repo / "uv.lock"
    if not lock.is_file():
        raise BundleError(f"no uv.lock in {repo}; the environment cannot be pinned")
    try:
        cfg = load_config(config_path)
    except Exception as exc:  # any load/validation failure is a refusal, with its reason
        raise BundleError(f"config {config_path} is invalid: {exc}") from exc
    if not cfg.experiment_id.strip():
        raise BundleError("experiment_id is empty")
    if not entry_point or not entry_point[0].strip():
        raise BundleError("entry_point is empty")
    hw = dict(hardware or {})
    unknown = sorted(set(hw) - set(HARDWARE_FIELDS))
    if unknown:
        raise BundleError(f"unknown hardware field(s): {', '.join(unknown)}")
    for line in [*hw.values(), *env_setup]:
        if "\n" in line or "\r" in line:
            raise BundleError(f"hardware/env value spans several lines: {line!r}")
    data: list[dict[str, Any]] = []
    for f in data_files:
        f = Path(f)
        if not f.is_file():
            raise BundleError(f"data file missing: {f}")
        if any(d["path"] == f.name for d in data):
            raise BundleError(f"data file basename not unique: {f.name}")
        data.append({"path": f.name, "bytes": f.stat().st_size, "sha256": sha256_file(f)})
    when = (now or datetime.now(UTC)).astimezone(UTC)
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    bundle_id = f"{cfg.experiment_id}-{commit[:8]}-{stamp}"
    target = Path(out_root) / bundle_id
    if target.exists():
        raise BundleError(f"bundle {target} already exists; never overwritten")

    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "experiment_id": cfg.experiment_id,
        "created_utc": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit,
        "git_dirty_source": False,
        "config": cfg.model_dump(),
        "config_sha256": _config_sha256(cfg),
        "seeds": {"seed": cfg.seed},
        "data": data,
        "uv_lock_sha256": sha256_file(lock),
        "hardware": {f: str(hw.get(f, UNSET)) for f in HARDWARE_FIELDS},
        "env_setup": list(env_setup),
        "entry_point": list(entry_point),
        "required_outputs": list(required_outputs),
        "checkpoints": [
            {"path": str(c["path"]), "external": bool(c.get("external", False))}
            for c in checkpoints
        ],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=out_root))
    try:
        (tmp / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        shutil.copyfile(config_path, tmp / "config.yaml")
        sbatch = tmp / "run.sbatch"
        sbatch.write_text(render_sbatch(bundle), encoding="utf-8")
        sbatch.chmod(0o755)
        write_manifest(tmp, BUNDLE_MANIFEST)
        os.rename(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def resolve_entry(bundle: Mapping[str, Any], config_path: Path, out_dir: Path) -> list[str]:
    """bundle['entry_point'] with `{config}` -> str(config_path), `{out}` -> str(out_dir)."""
    subs = {"{config}": str(config_path), "{out}": str(out_dir)}
    return [_subst(arg, subs) for arg in bundle["entry_point"]]


def _subst(arg: str, subs: Mapping[str, str]) -> str:
    for key, val in subs.items():
        arg = arg.replace(key, val)
    return arg


def _shell_arg(arg: str) -> str:
    """Shell word for one entry-point argv element. `{config}`/`{out}` become double-quoted
    shell variables; every other byte is literal."""
    if "{config}" not in arg and "{out}" not in arg:
        return shlex.quote(arg)
    parts = re.split(r"(\{config\}|\{out\})", arg)
    words = []
    for part in parts:
        if part == "{config}":
            words.append("$BUNDLE_DIR/config.yaml")
        elif part == "{out}":
            words.append("$OUT")
        else:
            words.append(re.sub(r'(["\\$`])', r"\\\1", part))
    return '"' + "".join(words) + '"'


def render_sbatch(bundle: Mapping[str, Any]) -> str:
    """Render templates/slurm.sh.j2 (str.format, no new deps) from a bundle dict.

    Each hardware field set to a real value -> `#SBATCH --<flag>=<value>` (job_name ->
    --job-name; scratch -> `SCRATCH_DIR=<value>` line, not an #SBATCH). A field equal to UNSET
    -> `# TODO(unset): <flag>` and NO active directive / assignment. env_setup lines are
    injected verbatim; empty -> a `# TODO(unset): no environment setup` comment. The entry
    point is shlex-quoted with `{config}` -> "$BUNDLE_DIR/config.yaml", `{out}` -> "$OUT".
    The output must contain no cluster-specific literal that did not come from `bundle`.
    """
    template = (Path(__file__).parent / "templates" / "slurm.sh.j2").read_text(encoding="utf-8")
    hw = bundle["hardware"]
    directives: list[str] = []
    scratch_line = ""
    for field in HARDWARE_FIELDS:
        value = str(hw.get(field, UNSET))
        if field == "scratch":
            scratch_line = (
                "# TODO(unset): scratch directory (SCRATCH_DIR); falls back to the bundle dir"
                if value == UNSET
                else f"SCRATCH_DIR={shlex.quote(value)}"
            )
            continue
        flag = {"job_name": "job-name", "cpus": "cpus-per-task"}.get(field, field)
        if value == UNSET:
            directives.append(f"# TODO(unset): --{flag}  (set for your cluster, or delete)")
        else:
            directives.append(f"#SBATCH --{flag}={value}")
    env = list(bundle.get("env_setup", []))
    env_setup = "\n".join(env) if env else (
        "# TODO(unset): no environment setup (add module/venv lines via env_setup)"
    )  # fmt: skip
    entry = " ".join(_shell_arg(a) for a in bundle["entry_point"])
    return template.format(
        bundle_id=bundle["bundle_id"],
        git_commit=bundle["git_commit"],
        sbatch_lines="\n".join(directives),
        scratch_line=scratch_line,
        env_setup=env_setup,
        entry=entry,
    )


def export_result(
    bundle_dir: Path,
    out_dir: Path,
    exit_code: int,
    *,
    started_utc: str,
    ended_utc: str,
    slurm_job_id: str | None = None,
) -> Path:
    """Worker-side: turn the job's `out_dir` into a self-describing result directory.

    out_dir already holds the job outputs plus stdout.log / stderr.log. This copies
    bundle.json + config.yaml in, scans every other file into result.json `outputs`
    ({path, bytes, sha256}), measures each declared checkpoint into `checkpoints`
    ({path, bytes, sha256, external, location}; external ones are hashed in place and NOT copied;
    a declared checkpoint that does not exist is omitted), sets status = "completed" iff
    exit_code == 0 else "failed", and writes RESULT.sha256. Returns out_dir.
    Refuses (BundleError) if out_dir already contains result.json (never overwrite a completed
    run). Does not read git, Research OS, or anything outside bundle_dir/out_dir/checkpoints.
    """
    bundle_dir, out_dir = Path(bundle_dir), Path(out_dir)
    if (out_dir / "result.json").exists():
        raise BundleError(f"{out_dir} already holds result.json; never overwritten")
    bundle_json = bundle_dir / "bundle.json"
    if not bundle_json.is_file() or not out_dir.is_dir():
        raise BundleError(f"bundle or output dir missing: {bundle_dir}, {out_dir}")
    try:
        bundle = json.loads(bundle_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleError(f"cannot read bundle.json: {exc}") from exc
    files, links = _regular_files(out_dir)
    if links:
        raise BundleError(f"output dir contains symlinks: {', '.join(sorted(links))}")
    for name in _BUNDLE_COPIES:
        shutil.copyfile(bundle_dir / name, out_dir / name)
    files, _ = _regular_files(out_dir)
    outputs = [
        {"path": rel, "bytes": (out_dir / rel).stat().st_size, "sha256": sha256_file(out_dir / rel)}
        for rel in sorted(files - set(_BUNDLE_COPIES) - {"result.json", RESULT_MANIFEST})
    ]
    checkpoints: list[dict[str, Any]] = []
    for ck in bundle.get("checkpoints", []):
        external = bool(ck["external"])
        src = Path(ck["path"]) if external else out_dir / ck["path"]
        if not src.is_file():
            continue  # omitted; verify reports it as not produced for a completed run
        checkpoints.append(
            {
                "path": ck["path"],
                "bytes": src.stat().st_size,
                "sha256": sha256_file(src),
                "external": external,
                "location": str(src) if external else ck["path"],
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle["bundle_id"],
        "experiment_id": bundle["experiment_id"],
        "git_commit": bundle["git_commit"],
        "config_sha256": bundle["config_sha256"],
        "status": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "slurm_job_id": slurm_job_id or None,
        "outputs": outputs,
        "checkpoints": checkpoints,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_manifest(out_dir, RESULT_MANIFEST)
    return out_dir
