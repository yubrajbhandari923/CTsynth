"""Compare two pipeline output dirs for exact reproducibility (NIfTI voxel data + zarr projections). usage: compare_runs.py DIR_A DIR_B"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path.home() / "ctsynth_truect/pylib"))
a, b = Path(sys.argv[1]), Path(sys.argv[2])
for f in sorted((a / "nifti").glob("*.nii.gz")):
    g = b / "nifti" / f.name
    if not g.exists():
        print(f"{f.name}: only in A"); continue
    if f.name == "phantom_material_labels.nii.gz":
        continue
    x, y = np.asarray(nib.load(f).dataobj), np.asarray(nib.load(g).dataobj)
    print(f"{f.name}: identical={np.array_equal(x, y)} max_abs_diff={float(np.abs(x.astype(np.float64) - y).max()):.3e} affine_equal={np.array_equal(nib.load(f).affine, nib.load(g).affine)}")
import zarr
za, zb = zarr.open(str(a / "projections/simulated_projections.zarr"), "r"), zarr.open(str(b / "projections/simulated_projections.zarr"), "r")
d = 0.0
for v0 in range(0, za.shape[0], 1000):
    d = max(d, float(np.abs(za[v0:v0 + 1000] - zb[v0:v0 + 1000]).max()))
print("projections.zarr", za.shape, "max_abs_diff", d, "identical", d == 0.0)
