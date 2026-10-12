"""Generate the small known-material XCIST dataset (paired-ct records). Run in the XCIST env.

`research run exec EXP-... -- .venv-xcist/bin/python projects/p04_xcist/generate.py OUT_DIR`
Materials (XCIST names): water, Ca, I. Truth = analytic disks rasterised to area-coverage volume
fractions; XCIST output is never used as truth. Phantom disks are stored in provenance so the
comparison can use exact analytic chords independent of voxelisation.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from ctsynth.paired.phantoms import Disk, fractions
from ctsynth.paired.schema import load_record, record_digest
from ctsynth.paired.xcist import XcistAcq, reexec_with_libgomp, run_xcist, to_paired_record

N, DX = 128, 2.0
XN = ["water", "Ca", "I"]
NAMES = ("water", "calcium", "iodine")  # CTsynth basis names for the same three materials

# per-material delta volume fractions inside each disk: (water, Ca, I)
PHANTOMS: dict[str, list[Disk]] = {
    "G1": [
        Disk(0, 0, 100.0, (1.0, 0.0, 0.0)),
        Disk(40, 30, 20.0, (-0.2, 0.2, 0.0)),
        Disk(-45, -20, 15.0, (-0.03, 0.0, 0.03)),
    ],
    "G2": [
        Disk(10, -5, 90.0, (1.0, 0.0, 0.0)),
        Disk(-30, 35, 12.0, (-0.1, 0.1, 0.0)),
        Disk(30, -40, 18.0, (-0.02, 0.0, 0.02)),
    ],
    "G3": [
        Disk(0, 0, 110.0, (1.0, 0.0, 0.0)),
        Disk(-50, 20, 10.0, (-0.05, 0.05, 0.0)),
        Disk(0, -50, 16.0, (-0.1, 0.1, 0.0)),
        Disk(55, 25, 22.0, (-0.3, 0.3, 0.0)),
    ],
}
BASE = XcistAcq(n_view=360, mA=800.0)
VARIANTS = {
    "ref": {},
    "sub": {"subsample": True},
    "noisy": {"quantum_noise": True, "electronic_noise": True, "mA": 20.0, "seed": 11},
}


def main(out_dir: Path) -> None:
    for pid, disks in PHANTOMS.items():
        f = fractions(disks, N, DX)
        for kvp in (80, 120):
            for var, over in VARIANTS.items():
                acq = replace(BASE, kvp=kvp, **over)
                work = out_dir / "_work" / f"{pid}_{kvp}_{var}"
                out = run_xcist(work, f, XN, DX, acq, recon=(var == "ref"))
                dens = np.array([out["spectrum"]["density_g_cm3"][m] for m in XN])
                to_paired_record(
                    out,
                    f,
                    NAMES,
                    dens,
                    acq,
                    phantom_id=pid,
                    acquisition_id=f"kvp{kvp}_{var}",
                    dest=out_dir / f"{pid}__kvp{kvp}_{var}",
                    attenuation_source="Geant4-EDLP97 (gecatsim GetMu)",
                    extra_provenance={
                        "phantom_disks": [d.__dict__ for d in disks],
                        "xcist_names": XN,
                    },
                )
                print("wrote", pid, kvp, var, flush=True)
    man = {p.name: record_digest(load_record(p)) for p in sorted(out_dir.glob("G*__*"))}
    (out_dir / "MANIFEST.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    reexec_with_libgomp()
    main(Path(sys.argv[1]))
