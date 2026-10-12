"""Read-only cluster environment probe (stdlib, py3.8-compatible): which interpreters/tools/modules exist."""
import json
import os
import shutil
import subprocess

run_dir = os.environ["RESEARCH_RUN_DIR"]


def sh(cmd):
    p = subprocess.run(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
    return p.stdout.strip()[:6000]


tools = ["python3", "python3.9", "python3.10", "python3.11", "python3.12", "python3.13", "uv", "conda", "mamba",
         "micromamba", "gcc", "g++", "cmake", "make", "git", "module", "apptainer", "singularity"]
res = {"which": {t: shutil.which(t) or sh("command -v %s" % t) or None for t in tools},
       "module_avail_python": sh("module avail python 2>&1 | head -60"),
       "module_avail_gcc_cuda_conda": sh("module avail gcc cuda conda anaconda miniconda 2>&1 | head -60"),
       "lmod_or_env": sh("echo ${LMOD_VERSION:-none}; echo $SHELL"),
       "cpu": sh("nproc; uname -m"), "home_quota_df": sh("df -h $HOME | tail -1")}
out = os.path.join(run_dir, "env_probe.json")
json.dump(res, open(out, "w"), indent=2, sort_keys=True)
with open(os.path.join(run_dir, "artifacts.jsonl"), "a") as f:
    f.write(json.dumps({"path": out, "name": "env_probe.json", "type": "json", "tags": ["probe"]}) + "\n")
with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
    f.write(json.dumps({"name": "probe_completed", "value": 1.0, "step": 0}) + "\n")
print(json.dumps(res, indent=2))
