#!/usr/bin/env bash
# Build an isolated XCIST env on the cluster, inside the project's approved directory only.
# Never touches system Python; uv cache and its Python downloads are redirected under $P.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/xcist_env_path.sh"
P="$CTSYNTH_XCIST_ENV"
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
  PY="$P/xcist-venv/bin/python"
  $UV pip install --python "$PY" "numpy<2" scipy matplotlib tqdm pydantic pyyaml > "$OUT/pip_deps.log" 2>&1
  echo "deps rc=$?"
  # gecatsim 1.6.9's sdist ships clib_build/BuildLinux64 with CRLF line endings, which bash cannot run
  # (upstream packaging defect). Fetch the sdist, verify it against PyPI's sha256, fix line endings
  # in the build scripts only, then install from that local copy.
  python3 - "$P/src" > "$OUT/sdist_fetch.log" 2>&1 <<'PYEOF'
import hashlib, json, sys, tarfile, urllib.request, pathlib
dest = pathlib.Path(sys.argv[1]); dest.mkdir(parents=True, exist_ok=True)
meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/gecatsim/1.6.9/json"))
u = next(x for x in meta["urls"] if x["packagetype"] == "sdist")
data = urllib.request.urlopen(u["url"]).read()
assert hashlib.sha256(data).hexdigest() == u["digests"]["sha256"], "sdist sha256 mismatch"
(dest / u["filename"]).write_bytes(data)
tarfile.open(dest / u["filename"]).extractall(dest)
print("sdist", u["filename"], "sha256 verified", u["digests"]["sha256"])
PYEOF
  cat "$OUT/sdist_fetch.log"
  SRC=$(ls -d "$P"/src/gecatsim-1.6.9 2>/dev/null)
  find "$SRC/gecatsim/clib_build" -type f \( -name "Build*" -o -name "Makefile*" -o -name "*.mk" \) \
      -exec sed -i 's/\r$//' {} + 
  echo "build path A: rebuild with converted line endings"
  $UV pip install -v --python "$PY" --no-deps "$SRC" > "$OUT/pip_install_full.log" 2>&1; rcA=$?
  echo "path A rc=$rcA"
  if [ $rcA -ne 0 ]; then
    grep -n -iE "error|fatal|undefined|cannot|No such" "$OUT/pip_install_full.log" | head -25
    echo "build path B: skip the rebuild and use the prebuilt Linux libraries shipped in the sdist"
    printf '#!/bin/bash\nexit 0\n' > "$SRC/gecatsim/clib_build/BuildLinux64"
    $UV pip install -v --python "$PY" --no-deps "$SRC" > "$OUT/pip_install_full_B.log" 2>&1; echo "path B rc=$?"
  fi
  "$P/xcist-venv/bin/python" -c "import sys,numpy,scipy,gecatsim;print('import ok',sys.version.split()[0],numpy.__version__,scipy.__version__,gecatsim.__file__)" 2>&1
  "$P/xcist-venv/bin/python" -c "import numpy as np; from gecatsim.pyfiles.GetMu import GetMu; print('C lib GetMu water 60keV mu/cm =', float(GetMu('water', np.array([60.0], dtype=np.single))[0]))" 2>&1 | tail -3
  $UV pip freeze --python "$P/xcist-venv/bin/python" 2>&1
} > "$OUT/env_setup.log" 2>&1
rc_import=$("$P/xcist-venv/bin/python" -c "import gecatsim" >/dev/null 2>&1; echo $?)
t1=$(date +%s)
printf '{"import_rc": %s, "seconds": %s, "env": "%s"}\n' "$rc_import" "$((t1-t0))" "$P/xcist-venv" > "$OUT/env_setup.json"
[ -f "$OUT/pip_install_full_B.log" ] && printf '{"path": "%s/pip_install_full_B.log", "name": "pip_install_full_B.log", "type": "log"}\n' "$(cd "$OUT" && pwd)" >> "$OUT/artifacts.jsonl"
printf '{"path": "%s/pip_install_full.log", "name": "pip_install_full.log", "type": "log"}\n' "$(cd "$OUT" && pwd)" >> "$OUT/artifacts.jsonl"
printf '{"path": "%s", "name": "env_setup.log", "type": "log"}\n{"path": "%s", "name": "env_setup.json", "type": "json"}\n' \
  "$(cd "$OUT" && pwd)/env_setup.log" "$(cd "$OUT" && pwd)/env_setup.json" >> "$OUT/artifacts.jsonl"
printf '{"name": "xcist_env_import_ok", "value": %s, "step": 0}\n' "$([ "$rc_import" = 0 ] && echo 1.0 || echo 0.0)" >> "$OUT/metrics.jsonl"
cat "$OUT/env_setup.log" | tail -40
[ "$rc_import" = 0 ]
