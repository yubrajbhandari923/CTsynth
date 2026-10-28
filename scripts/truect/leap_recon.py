"""Helical FBP of TrueCT DICOM-CT-PD projections with LLNL LEAP (curved-detector helical cone-beam, GPU).

Geometry (all from the DICOM-CT-PD tags / PDF Table 1): SOD 575 mm, SDD 1050 mm, 900 channels x 64 rows, 1 mm detector
pixels at the detector (channel = arc length at SDD, cylindrical), 1000 views/rotation, table advance 35.05 mm/rotation,
angle step 2*pi/1000, no flying focal spot. Detector centre assumed at the middle element (no tag gives it): UNVERIFIED.
Output unit: linear attenuation mu in 1/mm (input line integrals are dimensionless and voxels are in mm). Not HU.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/yb107/ctsynth_truect/leap/LEAP/src")
from leapctype import tomographicModels  # noqa: E402

from common import OUT  # noqa: E402

SOD, SDD, PIX_MM = 575.0, 1050.0, 1.0

def load_g(series, v0=0, v1=None, ch_flip=False, row_flip=False):
    d = OUT / "projections" / series
    raw = np.load(d / "raw_u16.npy", mmap_mode="r")
    m = np.load(d / "meta.npz")
    v1 = raw.shape[0] if v1 is None else v1
    g = raw[v0:v1].astype(np.float32) * m["slope"][v0:v1, None, None] + m["intercept"][v0:v1, None, None]  # (view, ch, row)
    g = g.astype(np.float32)
    g = np.ascontiguousarray(g.transpose(0, 2, 1))  # LEAP wants (view, row, col=channel)
    if ch_flip:
        g = np.ascontiguousarray(g[:, :, ::-1])
    if row_flip:
        g = np.ascontiguousarray(g[:, ::-1, :])
    ang = np.unwrap(m["angle_rad"][v0:v1].astype(np.float64))
    return g, ang, m["z_mm"][v0:v1].astype(np.float64)

def recon(series, nxy, vox_mm, dz_mm, nz=None, rot_sign=1, ch_flip=False, row_flip=False, v0=0, v1=None, z_center_mm=None, extra=None, col_shift=0.0, row_shift=0.0, off_x_mm=0.0, off_y_mm=0.0):
    """Return (volume[z,y,x] mu 1/mm, z_phys_mm of each slice, info). z_phys = table z (mm) in the DICOM-CT-PD axial-position frame."""
    g, ang, zmm = load_g(series, v0, v1, ch_flip, row_flip)
    n, rows, cols = g.shape
    pitch = float(np.polyfit(ang, zmm, 1)[0])  # mm/rad from the tags (z_mm vs unwrapped angle)
    phis = np.degrees(rot_sign * ang).astype(np.float32)
    L = tomographicModels()
    L.set_conebeam(n, rows, cols, PIX_MM, PIX_MM / 1.0, 0.5 * (rows - 1) + row_shift, 0.5 * (cols - 1) + col_shift, phis, SOD, SDD)
    L.set_curvedDetector()
    L.set_helicalPitch(rot_sign * pitch)
    zmid = 0.5 * (zmm[0] + zmm[-1])
    z_c = (zmid if z_center_mm is None else z_center_mm)
    if nz is None:
        nz = int(np.ceil((zmm[-1] - zmm[0]) / dz_mm))
    L.set_volume(nxy, nxy, nz, vox_mm, dz_mm, off_x_mm, off_y_mm, z_c - zmid)
    L.set_gpu(0) if hasattr(L, "set_gpu") else None
    f = L.allocateVolume()
    L.FBP(g, f)
    zphys = zmid + (z_c - zmid) + (np.arange(nz) - 0.5 * (nz - 1)) * dz_mm
    info = dict(pitch_mm_per_rad=pitch, n_views=n, nxy=nxy, vox_mm=vox_mm, dz_mm=dz_mm, nz=nz, rot_sign=rot_sign,
                ch_flip=ch_flip, row_flip=row_flip, z_scan_mm=[float(zmm[0]), float(zmm[-1])], vol_shape=list(f.shape))
    return f, zphys, info
