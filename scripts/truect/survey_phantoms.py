"""Survey phantoms: label histogram, label bounding box, and which C-order reshape gives spatially smooth data.

The filename dims are an ordered triple; a wrong axis order scrambles strides, so neighbouring voxels disagree.
Score = fraction of equal neighbours along each axis in a central block; the smooth reshape is the true layout.
"""
import itertools
import json
import sys

import numpy as np

from common import EXTRA, OUT, PHANTOMS

def smooth_scores(raw: np.ndarray, dims: tuple[int, int, int]) -> dict:
    out = {}
    for perm in sorted(set(itertools.permutations(dims))):
        vol = raw.reshape(perm)
        c = [s // 2 for s in perm]
        blk = np.asarray(vol[c[0] - 32:c[0] + 32, c[1] - 32:c[1] + 32, c[2] - 32:c[2] + 32])
        ag = [float((np.diff(blk, axis=a) == 0).mean()) for a in range(3)]
        out[str(perm)] = {"agree_per_axis": ag, "mean": float(np.mean(ag))}
    return out

def main() -> None:
    cases = {k: (v[0], v[1]) for k, v in PHANTOMS.items()} | {k: (v[0], v[1]) for k, v in EXTRA.items()}
    res = {}
    for name, (path, dims) in cases.items():
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        assert raw.size == int(np.prod(dims)), (name, raw.size, dims)
        hist = np.bincount(raw, minlength=256)
        res[name] = {
            "file": str(path), "dims_filename": dims,
            "labels_present": {int(i): int(n) for i, n in enumerate(hist) if n},
            "scores": smooth_scores(raw, dims),
        }
        print(name, {k: round(v["mean"], 3) for k, v in res[name]["scores"].items()}, flush=True)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports/survey_phantoms.json").write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
