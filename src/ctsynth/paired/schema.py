"""Versioned, simulator-independent record of one physical phantom + one CT acquisition.

A record is a directory: `record.json` (this schema) plus array files referenced by name and
sha256. Arrays stay outside Git. Producers (XCIST, DukeSim, mock) convert INTO this format.

Axis convention (binding): truth volumes are `(n_material, nz, ny, nx)`, i.e. `(iz, iy, ix)` per
material; `grid.affine` maps `[ix, iy, iz, 1]` voxel-centre indices to `[x, y, z, 1]` in mm in the
named scanner `frame`. 2D slices are `nz == 1`. Unknown calibration is declared in `unknown`, never
silently defaulted.
"""

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ctsynth.hashing import sha256_file

SCHEMA_VERSION = "paired-ct/0.1"
Semantics = Literal["volume_fraction", "partial_mass_density_g_cm3"]
Kind = Literal[
    "line_integral_prep", "counts_expected", "counts_noisy", "signal_air", "signal_scan", "recon_hu"
]
# fields a producer may legitimately be unable to supply; anything else missing is an error
UNKNOWABLE = {"spectrum", "detector", "scatter", "noise", "recon", "attenuation_source"}


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Grid:
    shape: tuple[int, int, int]  # (nz, ny, nx)
    affine: tuple[tuple[float, ...], ...]  # 4x4, index (ix, iy, iz) -> mm
    frame: str  # e.g. "xcist-scanner" ; the producer's own frame, named

    def spacing_mm(self) -> tuple[float, float, float]:
        """(dz, dy, dx) from affine column norms."""
        a = np.asarray(self.affine, dtype=float)
        return tuple(float(np.linalg.norm(a[:3, c])) for c in (2, 1, 0))  # type: ignore[return-value]

    def validate(self) -> None:
        a = np.asarray(self.affine, dtype=float)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise SchemaError(f"grid shape must be 3 positive ints, got {self.shape}")
        if a.shape != (4, 4) or not np.isfinite(a).all() or not np.allclose(a[3], [0, 0, 0, 1]):
            raise SchemaError("affine must be finite 4x4 with last row [0,0,0,1]")
        lin = a[:3, :3]
        norms = np.linalg.norm(lin, axis=0)
        if (norms <= 0).any() or not np.allclose(
            lin.T @ lin, np.diag(norms**2), atol=1e-6 * norms.max() ** 2
        ):
            raise SchemaError("affine axes must be orthogonal with positive spacing")
        if not self.frame:
            raise SchemaError("grid.frame must name the coordinate frame")


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    semantics: Semantics  # what the truth array holds; fraction != density
    ref_density_g_cm3: float | None  # required to convert fraction <-> density
    attenuation_source: str  # e.g. "NIST-XCOM", "Geant4-EDLP97", or "unknown"


@dataclass(frozen=True)
class Measurement:
    name: str
    kind: Kind
    file: str
    shape: tuple[int, ...]
    dtype: str
    axes: tuple[str, ...]  # e.g. ("view", "row", "col")
    units: str  # e.g. "1" (dimensionless -ln), "HU", "photons"
    energy_semantics: str  # e.g. "polychromatic-EI", "monochromatic-60keV", "bins:[...]"
    sha256: str


@dataclass(frozen=True)
class PairedRecord:
    record_id: str
    phantom_id: str  # split unit: all acquisitions of one phantom share it
    acquisition_id: str
    producer: dict[str, str]  # {"name","version", ...}
    grid: Grid
    materials: tuple[MaterialSpec, ...]
    truth_file: str  # (n_material, nz, ny, nx) float
    truth_sha256: str
    sum_constraint: Literal["le1", "eq1", "none"]  # for fractions
    acquisition: dict[
        str, Any
    ]  # geometry/spectrum/detector/scatter/noise/recon; values may be None
    unknown: tuple[str, ...]  # acquisition keys the producer could not supply
    measurements: tuple[Measurement, ...]
    provenance: dict[str, Any] = field(default_factory=dict)  # commits, versions, config hashes

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = json.loads(json.dumps(self, default=lambda o: o.__dict__))
        d["schema_version"] = SCHEMA_VERSION
        return d

    def validate(self) -> None:
        self.grid.validate()
        if not (self.record_id and self.phantom_id and self.acquisition_id):
            raise SchemaError("record_id, phantom_id and acquisition_id are mandatory")
        names = [m.name for m in self.materials]
        if not names or len(set(names)) != len(names):
            raise SchemaError("materials must be non-empty with unique names")
        for m in self.materials:
            if m.ref_density_g_cm3 is not None and m.ref_density_g_cm3 <= 0:
                raise SchemaError(f"{m.name}: ref_density must be > 0")
        if not self.measurements:
            raise SchemaError("at least one measurement is required")
        for key in self.unknown:
            if key not in UNKNOWABLE:
                raise SchemaError(f"'{key}' cannot be declared unknown")
        for key in UNKNOWABLE - {"attenuation_source"}:
            if self.acquisition.get(key) is None and key not in self.unknown:
                raise SchemaError(f"acquisition['{key}'] missing: supply it or list it in unknown")
            if self.acquisition.get(key) is not None and key in self.unknown:
                raise SchemaError(f"acquisition['{key}'] is both supplied and declared unknown")
        for m in self.materials:
            if m.attenuation_source == "unknown" and "attenuation_source" not in self.unknown:
                raise SchemaError(f"{m.name}: attenuation_source unknown but not declared")

    def validate_truth(self, truth: np.ndarray, tol: float = 1e-6) -> None:
        if truth.shape != (len(self.materials), *self.grid.shape):
            raise SchemaError(f"truth shape {truth.shape} != (n_mat, {self.grid.shape})")
        if not np.isfinite(truth).all() or (truth < -tol).any():
            raise SchemaError("truth must be finite and non-negative")
        frac = [m.semantics == "volume_fraction" for m in self.materials]
        if any(frac) and not all(frac):
            raise SchemaError("materials must share one semantics (fraction vs density)")
        if all(frac):
            s = truth.sum(axis=0)
            if (truth > 1 + tol).any() or (self.sum_constraint == "le1" and (s > 1 + tol).any()):
                raise SchemaError("volume fractions must be <= 1 and sum to <= 1")
            if self.sum_constraint == "eq1" and not np.allclose(s, 1, atol=1e-4):
                raise SchemaError("volume fractions must sum to 1")


def save_record(
    rec: PairedRecord, truth: np.ndarray, arrays: dict[str, np.ndarray], out: Path
) -> Path:
    """Write `out/` (must not exist). Hashes in `rec` must match; use `build_record` helpers."""
    rec.validate()
    rec.validate_truth(truth)
    out = Path(out)
    out.mkdir(parents=True)
    np.save(out / rec.truth_file, truth)
    for m in rec.measurements:
        a = arrays[m.name]
        if tuple(a.shape) != m.shape or str(a.dtype) != m.dtype:
            raise SchemaError(
                f"{m.name}: array {a.shape}/{a.dtype} != declared {m.shape}/{m.dtype}"
            )
        np.save(out / m.file, a)
    (out / "record.json").write_text(json.dumps(rec.to_json(), indent=2, sort_keys=True))
    load_record(out)  # round-trip verifies hashes
    return out


def npy_sha256(a: np.ndarray) -> str:
    """sha256 of the exact .npy bytes `np.save` writes for `a`."""
    buf = io.BytesIO()
    np.save(buf, a)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def load_record(path: Path, *, verify: bool = True) -> PairedRecord:
    path = Path(path)
    d = json.loads((path / "record.json").read_text())
    if d.pop("schema_version", None) != SCHEMA_VERSION:
        raise SchemaError(f"unsupported schema_version in {path}")
    g = d["grid"]
    rec = PairedRecord(
        **{
            **d,
            "grid": Grid(tuple(g["shape"]), tuple(tuple(r) for r in g["affine"]), g["frame"]),
            "materials": tuple(MaterialSpec(**m) for m in d["materials"]),
            "measurements": tuple(
                Measurement(**{**m, "shape": tuple(m["shape"]), "axes": tuple(m["axes"])})
                for m in d["measurements"]
            ),
            "unknown": tuple(d["unknown"]),
        }
    )
    rec.validate()
    if verify:
        for fname, want in [(rec.truth_file, rec.truth_sha256)] + [
            (m.file, m.sha256) for m in rec.measurements
        ]:
            if sha256_file(path / fname) != want:
                raise SchemaError(f"{fname}: sha256 mismatch (modified or corrupted)")
        rec.validate_truth(np.load(path / rec.truth_file))
    return rec


def record_digest(rec: PairedRecord) -> str:
    return hashlib.sha256(json.dumps(rec.to_json(), sort_keys=True).encode()).hexdigest()
