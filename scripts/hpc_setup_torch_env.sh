#!/usr/bin/env bash
# Build an isolated torch env on the cluster (inside the approved project dir only) and probe the GPU.
# Idempotent: skips the build when the env already imports torch. Pins torch to the uv.lock version.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/xcist_env_path.sh"
P="$CTSYNTH_XCIST_ENV"; OUT="${RESEARCH_RUN_DIR:-.}"
export UV_CACHE_DIR="$P/uv-cache" UV_PYTHON_INSTALL_DIR="$P/uv-python"
UV="$HOME/.local/bin/uv"; PY="$P/torch-venv/bin/python"
TORCH_VER=$(grep -A1 '^name = "torch"$' uv.lock | sed -n 's/^version = "\(.*\)"/\1/p' | head -1)
{
  echo "host=$(hostname) date=$(date -Is) torch_pin=$TORCH_VER"
  # node driver is CUDA 12.7: PyPI's default wheel is cu130 (cuda_available=False), so pin the cu126 build
  if ! "$PY" -c "import torch,sys;sys.exit(0 if torch.__version__.endswith('+cu126') else 1)" 2>/dev/null; then
    [ -x "$PY" ] || $UV venv --python 3.11 "$P/torch-venv"
    $UV pip install --python "$PY" "torch==$TORCH_VER" --index-url https://download.pytorch.org/whl/cu126 \
      --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match --reinstall-package torch && \
    $UV pip install --python "$PY" numpy pydantic pyyaml matplotlib scipy tqdm
  fi
  nvidia-smi -L 2>&1 | head -4
  "$PY" - <<'PYEOF'
import json, sys, torch
info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "python": sys.version.split()[0]}
print(json.dumps(info))
open("torch_env_probe.json", "w").write(json.dumps(info))
PYEOF
  $UV pip freeze --python "$PY"
} > "$OUT/torch_env_setup.log" 2>&1
tail -30 "$OUT/torch_env_setup.log"
cp torch_env_probe.json "$OUT/" 2>/dev/null
printf '{"path": "%s/torch_env_setup.log", "name": "torch_env_setup.log", "type": "log"}\n' "$(cd "$OUT" && pwd)" >> "$OUT/artifacts.jsonl"
printf '{"path": "%s/torch_env_probe.json", "name": "torch_env_probe.json", "type": "json"}\n' "$(cd "$OUT" && pwd)" >> "$OUT/artifacts.jsonl"
ok=$("$PY" -c "import torch;print(1.0 if torch.cuda.is_available() else 0.0)" 2>/dev/null || echo 0.0)
printf '{"name": "cuda_available", "value": %s, "step": 0}\n' "$ok" >> "$OUT/metrics.jsonl"
"$PY" -c "import torch" 2>/dev/null
