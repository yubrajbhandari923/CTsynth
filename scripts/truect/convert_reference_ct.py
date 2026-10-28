"""Reference reconstruction (192 Secondary-Capture DICOMs) -> NIfTI.

The DICOMs carry NO spatial tags (no ImagePositionPatient/PixelSpacing/RescaleSlope, empty InstanceNumber). So:
 - slice order = filename order (slice_001..192): direction along z UNVERIFIED;
 - in-plane pixel = PX_MM argument (PROVISIONAL; PDF says 500 mm FOV/512 and 0.5476 mm slices but this sample is 768 px). Measured
   against the phantom/LEAP: pixel 0.600 mm (body diameter ratio 1.0003) and slice spacing ~0.606 mm (80 mm body = 132 slices) -> 0.6 isotropic;
 - values assumed to be HU (int16, no rescale tags; min -1000 looks like a clip): UNVERIFIED until compared with a recon.
"""
import sys

import numpy as np
import pydicom

from common import OUT, REF
from niiutil import write_nii

px = float(sys.argv[1]) if len(sys.argv) > 1 else 500 / 768
axcodes = tuple(sys.argv[2]) if len(sys.argv) > 2 else ("S", "P", "L")
tag = sys.argv[3] if len(sys.argv) > 3 else "provisional"
dz = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
files = sorted((REF / "dcmrecon_reference").glob("slice_*.dcm"))
ds = [pydicom.dcmread(f) for f in files]
assert all(d.PixelRepresentation == 1 and d.BitsAllocated == 16 and "RescaleSlope" not in d for d in ds)
vol = np.stack([d.pixel_array for d in ds]).astype(np.int16)  # (slice, row, col)
print("shape", vol.shape, "HU-ish min/max", vol.min(), vol.max(), "files", len(files))
rep = write_nii(vol, (dz, px, px), axcodes,
                OUT / f"nifti/reference_ct_{tag}.nii.gz",
                f"TrueCT reference recon; HU ASSUMED; px={px:.4f} dz={dz:.4f}mm measured; axes {''.join(axcodes)} UNVERIFIED")
print(rep["orientation"], rep["voxel_mm"])
