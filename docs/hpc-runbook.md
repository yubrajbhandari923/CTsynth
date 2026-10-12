# HPC runbook — copyable commands

The laptop is authoritative. The HPC server is an **execution worker**: it never edits source and
never touches Research OS. Transfer is manual, by you. Nothing in CTsynth connects to a cluster.

Set these once per shell (examples — substitute your own; nothing here is a verified Duke value):

```bash
export HPC=user@cluster.example.edu          # your login node
export HPC_SCRATCH=/scratch/$USER/ctsynth    # a path YOU have verified exists
```

---

## 1. Laptop — propose and freeze

```bash
# the experiment must already exist in Research OS and be linked to a question/task
.research/bin/research experiment create "..." -q Q-00X --hypothesis "..." --method "..."

# commit first: prepare REFUSES if src/tests/configs are dirty
git status --porcelain -- src tests configs     # must be empty
uv run ctsynth experiment prepare configs/<name>.yaml --experiment EXP-00X --out bundles/
```

`prepare` writes `bundles/<EXP>-<sha7>-<utc>/` containing `bundle.json` (commit, resolved config +
hash, seeds, data manifest with checksums, `uv.lock` hash, hardware request, entry point, expected
outputs), the config copy, `run.sbatch`, and `MANIFEST.sha256`.

**Fill in the SLURM fields it marks `TODO`** (partition, account, time, GPUs, module lines). They are
deliberately left unset: guessing a cluster's partition silently produces a job that never runs.

## 2. Transfer to the worker

```bash
rsync -av bundles/<BUNDLE_ID>/ "$HPC:$HPC_SCRATCH/bundles/<BUNDLE_ID>/"
```

## 3. Worker — check out the pinned commit and run

```bash
ssh "$HPC"
cd "$HPC_SCRATCH"
git clone <repo-url> src-<BUNDLE_ID> && cd src-<BUNDLE_ID>
git checkout <GIT_COMMIT_FROM_bundle.json>    # exact SHA; never a branch name
uv sync --locked                              # must match bundle.json uv_lock_sha256
cd "$HPC_SCRATCH/bundles/<BUNDLE_ID>"
sbatch run.sbatch
```

**Do not edit code on the worker.** A dirty source tree makes the result non-citable
(`git_dirty_source`, see `docs/hpc-workflow.md`).

## 4. Retrieve results

```bash
rsync -av "$HPC:$HPC_SCRATCH/results/<RUN_DIR>/" ./incoming/<RUN_DIR>/
```

Large checkpoints marked `external` in the bundle stay on the cluster; only their path and hash are
imported. Record where you left them.

## 5. Laptop — verify, import, report

```bash
uv run ctsynth experiment verify ./incoming/<RUN_DIR>        # hashes, commit, config, outputs
uv run ctsynth experiment import ./incoming/<RUN_DIR>        # registers via `research run attach`
uv run ctsynth experiment report <RUN_ID>                    # markdown summary
```

`verify` never executes anything from the directory it checks. `import` refuses a second import of
the same run and never overwrites an existing one.

## 6. Opus interprets

An imported run is **evidence, not a conclusion**. Promotion to a finding requires the gate in
`CLAUDE.md` (D-008): bound+verified metrics, independent review, explicit scope.

```bash
.research/bin/research metric add <name> <value> --run <RUN_ID> --unit <unit>
.research/bin/research finding create "..." --supports EXP-00X <RUN_ID>
.research/bin/research evidence bind F-0XX --run <RUN_ID> --metric <name>
.research/bin/research evidence verify F-0XX
```

---

## Failure modes and what they mean
| Symptom | Cause | Fix |
|---|---|---|
| `prepare` refuses | `src/tests/configs` dirty | commit first; a dirty run is not citable |
| `verify` hash mismatch | file corrupted or edited in transit | re-transfer; never import a mismatched bundle |
| `verify` unknown commit | worker ran code not in this repo | push the commit, or the result is untraceable |
| `import` says already imported | duplicate | intentional: completed runs are never overwritten |
| job never starts | unfilled `TODO` SLURM fields | fill partition/account/time for YOUR cluster |
