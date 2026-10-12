# HPC experiment-bundle spec (T-030/T-031)

Implements `docs/aapm/hpc-handoff.md`. Interfaces: `src/ctsynth/hpc/`. Frozen tests: `tests/hpc/`.
Rules: laptop is authoritative; the worker only runs and never edits source or Research OS;
transfer is manual (no SSH/network code anywhere); Research OS (`research run attach`) is the
only run registry.

## Flow
`ctsynth prepare` (laptop) -> human copies the bundle dir out -> worker checks out the pinned commit
(`git archive`, no edits), submits `run.sbatch` -> job runs, `ctsynth export` seals the result dir ->
human copies it back -> `ctsynth verify` -> `ctsynth import` -> `ctsynth report`.

## bundle.json (schema_version 1)
`schema_version`, `bundle_id` (`<EXP>-<sha8>-<UTC %Y%m%dT%H%M%SZ>`, also the dir name),
`experiment_id`, `created_utc`, `git_commit` (full 40-hex), `git_dirty_source` (must be false or prepare
refuses; same pathspec as `runmeta`: src tests configs, untracked included), `config` (resolved
`ExperimentConfig.model_dump()`), `config_sha256` (sha256 of `model_dump_json()`, identical to
`runmeta.collect()`), `seeds` (`{"seed": cfg.seed}`), `data` (`[{path, bytes, sha256}]`; `path` is the
basename, unique; data is NOT in the bundle, the worker supplies it and `verify --data-root` checks it),
`uv_lock_sha256`, `hardware` (`partition account gpus cpus mem time job_name scratch`, default
`"unset"`), `env_setup` (list of shell lines), `entry_point` (argv with `{config}` `{out}`; cwd is the
source root), `required_outputs` (filenames), `checkpoints` (`[{path, external}]` declared; sizes and
hashes are measured at export, so they live in result.json).
Bundle dir: `bundle.json`, `config.yaml` (byte copy), `run.sbatch`, `MANIFEST.sha256`
(`sha256sum` format, covers every other file). Written in a temp dir then renamed; an existing dir is
never overwritten.

## result dir (written by `export`)
Job outputs + `stdout.log` + `stderr.log` + copies of `bundle.json`/`config.yaml` + `result.json` +
`RESULT.sha256`. result.json: `schema_version bundle_id experiment_id git_commit config_sha256
status` (`completed` iff exit 0 else `failed`) `exit_code started_utc ended_utc slurm_job_id outputs`
(`[{path,bytes,sha256}]`) `checkpoints` (`[{path,bytes,sha256,external,location}]`; external files are
hashed in place and never copied). A failed run needs none of its required outputs.

## verify
Collects all problems, exit 1 if any. Checks: manifest complete and hashes match (unlisted files,
absolute / `..` / symlink paths rejected); schema version; config re-loaded and its hash equals
`config_sha256` (catches a forged manifest); `git_dirty_source` false; commit is full-length (a prefix
is "ambiguous") and `git rev-parse --verify <sha>^{commit}` returns it ("unknown" otherwise; skipped,
not passed, when no `--repo`, e.g. on the worker); result-vs-bundle agreement; status/exit-code
consistency; outputs/checkpoints present; completed runs have every required output. Inert on hostile
dirs: JSON/YAML/bytes only, argv-list `git` with the sha pre-validated, nothing executed or imported
from the directory.

## import
verify -> idempotency marker `runs_root/.imported/<sha256 of RESULT.sha256>.json` -> copy to
`.staging-<fp>` (external checkpoints are not copied) -> `research run attach` (argv list, `--json`,
optional `--root`; id parsed from JSON) -> rename to `runs_root/<RUN-id>/` -> write marker. Refuses
and cleans up on any failure; never overwrites. Failed jobs import with `--status failed`.
Known limit: if `runs_root/<RUN-id>` exists after attach succeeded, import refuses but the run is
already registered in Research OS (Research OS ids are unique, so this means tampering). A result
from an old commit imports as-is; the commit is recorded, ancestry of HEAD is not required.
`runs_root` (conventionally `runs_imported/`) must be gitignored.

## SLURM template
`templates/slurm.sh.j2` is a `str.format` template (no Jinja). No cluster literals anywhere; every
unset field renders `# TODO(unset): <flag>` and no directive.

## Tests
Two temp dirs simulate the boundary; `FakeAttach` replaces `research run attach`; one test drives the
CLI against a throwaway `research init` root via `--root` (never the real `.research/`).
