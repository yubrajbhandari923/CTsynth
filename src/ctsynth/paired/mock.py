"""Mock producer: analytic disk phantoms, exact parallel-beam chords, Beer-Lambert counts.

Deliberately torch-free and independent of ctsynth.physics so it can serve as an inverse-crime-free
reference in tests. It is a fixture, not a scanner: no scatter, detector blur or reconstruction.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ctsynth.paired.schema import (
    Grid,
    MaterialSpec,
    Measurement,
    PairedRecord,
    npy_sha256,
    save_record,
)
from ctsynth.seeding import rng


@dataclass(frozen=True)
class Disk:
    material: int
    cx_cm: float
    cy_cm: float
    r_cm: float


def _grid(n: int, dx_cm: float) -> Grid:
    mm = dx_cm * 10.0
    off = -(n - 1) / 2 * mm
    aff = ((mm, 0, 0, off), (0, mm, 0, off), (0, 0, mm, 0), (0, 0, 0, 1))
    return Grid((1, n, n), aff, "ctsynth-mock")  # (ix,iy,iz)->mm, image centred at origin


def disk_chords(disks: list[Disk], n_mat: int, angles: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Exact line integrals of path length (cm) per material: (n_mat, n_view, n_det)."""
    out = np.zeros((n_mat, len(angles), len(s)))
    for d in disks:
        for v, th in enumerate(angles):
            off = s - (-d.cx_cm * np.sin(th) + d.cy_cm * np.cos(th))  # signed distance to centre
            out[d.material, v] += 2 * np.sqrt(np.clip(d.r_cm**2 - off**2, 0, None))
    return out


def make_mock_record(
    out: Path,
    *,
    seed: int,
    phantom_id: str,
    acquisition_id: str,
    energies_keV: np.ndarray,
    fluence: np.ndarray,
    kappa: np.ndarray,
    names: tuple[str, ...],
    ref_density: np.ndarray,
    n: int = 32,
    dx_cm: float = 0.5,
    n_view: int = 16,
    photons_scale: float = 1.0,
    noisy: bool = True,
) -> Path:
    """Random disks (from `seed`), fraction-valued truth; writes a record directory."""
    gen = rng(seed)
    n_mat = len(names)
    half = n * dx_cm / 2
    # three disks on a ring at 120 degrees: centres 0.5*half out, radii <= 0.25*half, so they
    # cannot overlap and stay inside the FOV (additive chords then equal the pixel-labelled truth)
    phase = float(gen.uniform(0, 2 * np.pi))
    disks = [
        Disk(
            int(gen.integers(n_mat)),
            float(0.5 * half * np.cos(phase + 2 * np.pi * k / 3)),
            float(0.5 * half * np.sin(phase + 2 * np.pi * k / 3)),
            float(gen.uniform(0.15, 0.25) * half),
        )
        for k in range(3)
    ]
    # truth: pixel-centre labelling (fractions in {0,1}); edge pixels differ from the exact chords
    ax = (np.arange(n) - (n - 1) / 2) * dx_cm
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    truth = np.zeros((n_mat, 1, n, n))
    for d in disks:
        inside = (xx - d.cx_cm) ** 2 + (yy - d.cy_cm) ** 2 <= d.r_cm**2
        truth[:, 0][:, inside] = 0.0
        truth[d.material, 0][inside] = 1.0
    angles = np.linspace(0, np.pi, n_view, endpoint=False)
    s = (np.arange(n) - (n - 1) / 2) * dx_cm
    path = disk_chords(disks, n_mat, angles, s)  # cm of each material (fraction 1)
    a = path * ref_density[:, None, None]  # g/cm^2
    expo = np.einsum("mvd,me->evd", a, kappa)
    counts = (fluence * photons_scale)[:, None, None] * np.exp(-expo)
    expected = counts.sum(0)
    arrays = {"expected_counts": expected.astype(np.float64)}
    if noisy:
        arrays["noisy_counts"] = gen.poisson(expected).astype(np.float64)
    meas = tuple(
        Measurement(
            name=k,
            kind="counts_expected" if "expected" in k else "counts_noisy",
            file=f"{k}.npy",
            shape=v.shape,
            dtype=str(v.dtype),
            axes=("view", "det"),
            units="photons",
            energy_semantics="polychromatic-EI",
            sha256=npy_sha256(v),
        )
        for k, v in arrays.items()
    )
    rec = PairedRecord(
        record_id=f"{phantom_id}:{acquisition_id}",
        phantom_id=phantom_id,
        acquisition_id=acquisition_id,
        producer={"name": "ctsynth-mock", "version": "1"},
        grid=_grid(n, dx_cm),
        materials=tuple(
            MaterialSpec(nm, "volume_fraction", float(r), "NIST-XCOM")
            for nm, r in zip(names, ref_density, strict=True)
        ),
        truth_file="truth.npy",
        truth_sha256=npy_sha256(truth),
        sum_constraint="le1",
        acquisition={
            "geometry": {
                "type": "parallel-2d",
                "n_view": n_view,
                "n_det": n,
                "det_spacing_cm": dx_cm,
            },
            "spectrum": {"energies_keV": energies_keV.tolist(), "photons_scale": photons_scale},
            "detector": {"type": "ideal-photon-integrating"},
            "scatter": None,
            "noise": {"poisson": noisy, "seed": seed},
            "recon": None,
        },
        unknown=("scatter", "recon"),
        measurements=meas,
        provenance={"seed": seed},
    )
    return save_record(rec, truth, arrays, out)
