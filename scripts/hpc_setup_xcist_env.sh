#!/usr/bin/env bash
# Build an isolated XCIST env on the cluster, inside the project's approved directory only.
# Never touches system Python; uv cache and its Python downloads are redirected under $P.
set -uo pipefail
P=/home/yb107/research-os/projects/CTsynth/env
OUT="${RESEARCH_RUN_DIR:-.}"
mkdir -p "$P"
export UV_CACHE_DIR="$P/uv-cache" UV_PYTHON_INSTALL_DIR="$P/uv-python"
UV="$HOME/.local/bin/uv"
t0=$(date +%s)
{
  echo "host=$(hostname) date=$(date -Is)"
  echo "uv=$($UV --version 2>&1)"
  echo "system gcc: $(gcc --version 2>&1 | head -1)"
  # best effort: newer compiler via environment modules, needed only if gecatsim rebuilds its lib
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load gcc/11.2.0 2>&1 || echo "module gcc/11.2.0 not loaded"
  echo "gcc now: $(gcc --version 2>&1 | head -1)"
  $UV venv --python 3.11 "$P/xcist-venv" 2>&1; echo "uv venv rc=$?"
  $UV pip install -v --python "$P/xcist-venv/bin/python" "numpy<2" scipy matplotlib tqdm pydantic \
      pyyaml gecatsim==1.6.9 > "$OUT/pip_install_full.log" 2>&1; echo "uv pip rc=$?"
  grep -n -iE "error|fatal|undefined|not found|cannot|No such" "$OUT/pip_install_full.log" | head -40
  "$P/xcist-venv/bin/python" -c "import sys,numpy,scipy,gecatsim;print('import ok',sys.version.split()[0],numpy.__version__,scipy.__version__,gecatsim.__file__)" 2>&1
  $UV pip freeze --python "$P/xcist-venv/bin/python" 2>&1
} > "$OUT/env_setup.log" 2>&1
rc_import=$("$P/xcist-venv/bin/python" -c "import gecatsim" >/dev/null 2>&1; echo $?)
t1=$(date +%s)
printf '{"import_rc": %s, "seconds": %s, "env": "%s"}\n' "$rc_import" "$((t1-t0))" "$P/xcist-venv" > "$OUT/env_setup.json"
printf '{"path": "%s/pip_install_full.log", "name": "pip_install_full.log", "type": "log"}\n' "$(cd "$OUT" && pwd)" >> "$OUT/artifacts.jsonl"
printf '{"path": "%s", "name": "env_setup.log", "type": "log"}\n{"path": "%s", "name": "env_setup.json", "type": "json"}\n' \
  "$(cd "$OUT" && pwd)/env_setup.log" "$(cd "$OUT" && pwd)/env_setup.json" >> "$OUT/artifacts.jsonl"
printf '{"name": "xcist_env_import_ok", "value": %s, "step": 0}\n' "$([ "$rc_import" = 0 ] && echo 1.0 || echo 0.0)" >> "$OUT/metrics.jsonl"
cat "$OUT/env_setup.log" | tail -40
[ "$rc_import" = 0 ]
