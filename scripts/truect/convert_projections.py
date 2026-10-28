"""Decode a DICOM-CT-PD projection series into native-scale arrays under outputs/projections/<series>/.

Stored: raw_u16.npy (views, 900 channels, 64 rows) exactly as in the DICOM pixel data, plus per-view
rescale slope/intercept (value = slope*raw + intercept = post-log, beam-hardening-corrected line integral, unitless),
tube angle (rad), table z (mm) and the mAs vector. Nothing is normalised.
"""
import json
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pydicom

from common import OUT, PROJECTIONS

def read_one(path: str):
    d = pydicom.dcmread(path)
    raw = np.frombuffer(d.PixelData, dtype="<u2").reshape(int(d.Rows), int(d.Columns))
    f32 = lambda t: struct.unpack("<f", bytes(d[t].value))[0]
    return (int(d.InstanceNumber), raw, float(d.RescaleSlope), float(d.RescaleIntercept),
            f32((0x7031, 0x1001)), f32((0x7031, 0x1002)))

def main(series: str, workers: int) -> None:
    src = PROJECTIONS[series]
    files = sorted(src.glob("proj_*.dcm"))
    out = OUT / "projections" / series
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        recs = sorted(ex.map(read_one, map(str, files), chunksize=64), key=lambda r: r[0])
    inst = np.array([r[0] for r in recs])
    assert (inst == np.arange(1, len(recs) + 1)).all(), "instance numbers not contiguous"
    raw = np.stack([r[1] for r in recs])
    np.save(out / "raw_u16.npy", raw)
    meta = dict(slope=np.array([r[2] for r in recs]), intercept=np.array([r[3] for r in recs]),
                angle_rad=np.array([r[4] for r in recs]), z_mm=np.array([r[5] for r in recs]))
    mas = sorted(src.glob("mAs_vector_*.bin"))
    if mas:
        meta["mAs"] = np.fromfile(mas[0], dtype="<f4")
    np.savez(out / "meta.npz", **meta)
    ang, z = meta["angle_rad"], meta["z_mm"]
    summary = {
        "series": series, "source": str(src), "n_views": len(recs), "raw_shape_views_ch_rows": list(raw.shape),
        "raw_dtype": "uint16 (PixelRepresentation 0)", "raw_min": int(raw.min()), "raw_max": int(raw.max()),
        "slope_range": [float(meta["slope"].min()), float(meta["slope"].max())],
        "intercept_range": [float(meta["intercept"].min()), float(meta["intercept"].max())],
        "angle_total_rad": float(ang[-1] - ang[0]), "angle_step_rad_first": float(ang[1] - ang[0]),
        "z_step_mm_first": float(z[1] - z[0]), "z_total_mm": float(z[-1] - z[0]),
        "angle_is_wrapped": bool(np.any(np.diff(ang) < 0)),
        "mAs": None if "mAs" not in meta else [int(meta["mAs"].size), float(meta["mAs"].min()), float(meta["mAs"].max())],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
