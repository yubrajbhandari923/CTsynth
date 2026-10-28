"""NIfTI writing with an explicit axis-code -> RAS affine. Axis codes name the direction the array index INCREASES toward."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

_W = {"R": (0, 1), "L": (0, -1), "A": (1, 1), "P": (1, -1), "S": (2, 1), "I": (2, -1)}

def affine_from_axcodes(shape, spacing_mm, axcodes):
    A = np.eye(4)
    A[:3, :3] = 0
    for i, c in enumerate(axcodes):
        w, s = _W[c]
        A[w, i] = s * spacing_mm[i]
    A[:3, 3] = -A[:3, :3] @ ((np.array(shape) - 1) / 2.0)  # volume centred on the origin (no physical origin known)
    return A

def write_nii(arr, spacing_mm, axcodes, path: Path, description: str):
    img = nib.Nifti1Image(arr, affine_from_axcodes(arr.shape, spacing_mm, axcodes))
    img.header["descrip"] = description.encode()[:79]
    img.set_qform(img.affine, 1)
    img.set_sform(img.affine, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, path)
    chk = nib.load(path)
    rep = {"file": str(path), "shape": list(chk.shape), "dtype": str(chk.get_data_dtype()),
           "voxel_mm": [float(v) for v in chk.header.get_zooms()], "orientation": "".join(nib.aff2axcodes(chk.affine)),
           "array_axis_directions": list(axcodes), "affine": chk.affine.round(5).tolist(), "note": description}
    path.with_suffix("").with_suffix(".json").write_text(json.dumps(rep, indent=1))
    return rep
