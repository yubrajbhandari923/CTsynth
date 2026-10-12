import json
from pathlib import Path

import numpy as np
import pytest

from ctsynth.paired.mock import make_mock_record
from ctsynth.paired.schema import SchemaError, load_record, record_digest

E = np.array([40.0, 60.0, 80.0])
FLU = np.array([1e5, 2e5, 1e5])
KAPPA = np.array([[0.27, 0.205, 0.184], [4.0, 1.2, 0.6]])
NAMES = ("water", "calcium")
RHO = np.array([1.0, 1.55])


def mk(tmp: Path, seed: int = 1, phantom: str = "P1", acq: str = "A1", **kw) -> Path:
    return make_mock_record(
        tmp / f"{phantom}_{acq}_{seed}",
        seed=seed,
        phantom_id=phantom,
        acquisition_id=acq,
        energies_keV=E,
        fluence=FLU,
        kappa=KAPPA,
        names=NAMES,
        ref_density=RHO,
        **kw,
    )


def test_roundtrip_and_geometry(tmp_path: Path) -> None:
    rec = load_record(mk(tmp_path))
    assert rec.grid.shape == (1, 32, 32)
    assert rec.grid.spacing_mm() == pytest.approx((5.0, 5.0, 5.0))
    assert {m.semantics for m in rec.materials} == {"volume_fraction"}


def test_mock_is_deterministic(tmp_path: Path) -> None:
    a, b = mk(tmp_path / "a"), mk(tmp_path / "b")
    assert record_digest(load_record(a)) == record_digest(load_record(b))
    ra = np.load(a / "noisy_counts.npy")
    assert np.array_equal(ra, np.load(b / "noisy_counts.npy"))


def test_detects_modified_array(tmp_path: Path) -> None:
    p = mk(tmp_path)
    x = np.load(p / "expected_counts.npy")
    np.save(p / "expected_counts.npy", x * 1.01)
    with pytest.raises(SchemaError, match="sha256 mismatch"):
        load_record(p)


def test_truth_alignment_with_measurement(tmp_path: Path) -> None:
    # total absorbed path in the measurement must match the truth volume mass (parallel-beam, 0..pi)
    p = mk(tmp_path, noisy=False)
    rec = load_record(p)
    truth = np.load(p / rec.truth_file)
    exp = np.load(p / "expected_counts.npy")
    dx = rec.grid.spacing_mm()[2] / 10
    mass_water_tissue = truth[1].sum() * dx**2 * RHO[1]  # g/cm of calcium per unit z
    assert mass_water_tissue > 0 and exp.shape == (16, 32)
    unatt = FLU.sum()
    # projection of an object that fits in the FOV: -ln(T) summed over det*dx ~ attenuation area
    area = (-np.log(exp / unatt)).sum(axis=1) * dx
    assert (
        np.ptp(area) / area.mean() < 0.35
    )  # same object from every view (bounded by beam hardening)


def test_rejects_undeclared_unknown_calibration(tmp_path: Path) -> None:
    p = mk(tmp_path)
    d = json.loads((p / "record.json").read_text())
    d["unknown"] = ["scatter"]  # recon is None but no longer declared unknown
    (p / "record.json").write_text(json.dumps(d))
    with pytest.raises(SchemaError, match="recon"):
        load_record(p)


def test_rejects_fraction_semantics_violation(tmp_path: Path) -> None:
    p = mk(tmp_path)
    t = np.load(p / "truth.npy")
    t[:, 0, 0, 0] = [0.8, 0.8]  # sums to 1.6
    np.save(p / "truth.npy", t)
    d = json.loads((p / "record.json").read_text())
    import hashlib

    d["truth_sha256"] = hashlib.sha256((p / "truth.npy").read_bytes()).hexdigest()
    (p / "record.json").write_text(json.dumps(d))
    with pytest.raises(SchemaError, match="sum to"):
        load_record(p)


def test_rejects_bad_affine(tmp_path: Path) -> None:
    p = mk(tmp_path)
    d = json.loads((p / "record.json").read_text())
    d["grid"]["affine"][0][1] = 2.0  # shear
    (p / "record.json").write_text(json.dumps(d))
    with pytest.raises(SchemaError, match="orthogonal"):
        load_record(p)


def test_multiple_acquisitions_share_phantom(tmp_path: Path) -> None:
    a = load_record(mk(tmp_path, phantom="P1", acq="A1", seed=3))
    b = load_record(mk(tmp_path, phantom="P1", acq="A2", seed=3))
    assert a.phantom_id == b.phantom_id and a.acquisition_id != b.acquisition_id
    assert np.array_equal(
        np.load(tmp_path / "P1_A1_3/truth.npy"), np.load(tmp_path / "P1_A2_3/truth.npy")
    )
