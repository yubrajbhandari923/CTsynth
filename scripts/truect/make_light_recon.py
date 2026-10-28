"""Download-friendly copies of the HU-estimate reconstructions: rounded to int16 HU (no rescale; +-0.5 HU) and cropped to the
body bounding box + 15 voxels. The cropped affine is the original one with the index offset applied (geometry preserved)."""
import sys

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from common import OUT

for name in sys.argv[1:]:
    img = nib.load(OUT / f"nifti/{name}_HUest_refcal.nii.gz")
    hu = np.asarray(img.dataobj)
    lab, n = ndi.label(hu[::4, ::4, ::4] > -500)
    big = np.argwhere(lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)) * 4
    lo = np.maximum(big.min(0) - 15, 0); hi = np.minimum(big.max(0) + 16, hu.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    out = nib.Nifti1Image(np.clip(np.rint(hu[sl]), -32768, 32767).astype(np.int16), img.slicer[sl].affine)
    out.header["descrip"] = b"HU ESTIMATE int16 cropped; mu_water from reference case; frame IPL PROVISIONAL"
    out.set_qform(out.affine, 1); out.set_sform(out.affine, 1)
    p = OUT / f"nifti/{name}_HUest_refcal_int16_cropped.nii.gz"
    nib.save(out, p)
    chk = nib.load(p)
    print(name, chk.shape, "".join(nib.aff2axcodes(chk.affine)), chk.header.get_zooms(), p.stat().st_size // 2**20, "MB")
