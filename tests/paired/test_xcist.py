"""Runs real XCIST; skipped unless gecatsim is installed (use .venv-xcist). ~1 min total."""

from pathlib import Path

import numpy as np
import pytest

xc = pytest.importorskip("gecatsim")

from gecatsim.pyfiles.GetMu import GetMu  # noqa: E402

from ctsynth.paired.phantoms import Disk, chords, fractions  # noqa: E402
from ctsynth.paired.schema import load_record  # noqa: E402
from ctsynth.paired.xcist import XcistAcq, fan_rays, run_xcist, to_paired_record  # noqa: E402

N, DX = 128, 2.0
NAMES = ["water", "bone"]
E0 = 60.0
MU = {m: float(GetMu(m, np.array([E0], dtype=np.single))[0]) / 10 for m in NAMES}  # 1/mm
ACQ = XcistAcq(kvp=None, mono_keV=E0, n_view=72)


def _two_disks() -> list[Disk]:
    # asymmetric on purpose: a flip or transpose of the array changes every projection
    return [Disk(24.0, 40.0, 28.0, (1.0, 0.0)), Disk(-46.0, -64.0, 18.0, (0.0, 1.0))]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    d = _two_disks()
    return run_xcist(tmp_path_factory.mktemp("x") / "w", fractions(d, N, DX), NAMES, DX, ACQ)


def _pred(disks: list[Disk]) -> np.ndarray:
    src, det = fan_rays(ACQ)
    c = chords(disks, src, det)
    return MU["water"] * c[0] + MU["bone"] * c[1]


def test_orientation_and_calibration_vs_analytic(run: dict) -> None:
    prep = run["prep"]
    rel = np.sqrt(np.mean((_pred(_two_disks()) - prep) ** 2)) / prep.max()
    assert rel < 0.01  # observed ~0.3-0.7%: pixel partial volume + detector aperture
    d = _two_disks()
    wrong = {
        "y-flipped": [Disk(x.cx_mm, -x.cy_mm, x.r_mm, x.delta) for x in d],
        "x-flipped": [Disk(-x.cx_mm, x.cy_mm, x.r_mm, x.delta) for x in d],
        "transposed": [Disk(x.cy_mm, x.cx_mm, x.r_mm, x.delta) for x in d],
    }
    for name, w in wrong.items():
        r = np.sqrt(np.mean((_pred(w) - prep) ** 2)) / prep.max()
        assert r > 5 * rel, name


def test_mono_central_ray_is_beer_lambert_of_diameter(tmp_path: Path) -> None:
    out = run_xcist(
        tmp_path / "w", fractions([Disk(0, 0, 60.0, (1.0,))], N, DX), ["water"], DX, ACQ
    )
    p0 = out["prep"][0]
    j = np.argmax(p0)
    assert p0[j] == pytest.approx(MU["water"] * 120.0, rel=0.01)  # independent slab reference


def test_determinism(tmp_path: Path) -> None:
    f = fractions([Disk(0, 0, 50.0, (1.0,))], 64, DX)
    quiet = XcistAcq(kvp=120, n_view=24)
    n1 = run_xcist(tmp_path / "q1", f, ["water"], DX, quiet)["prep"]
    n2 = run_xcist(tmp_path / "q2", f, ["water"], DX, quiet)["prep"]
    assert np.array_equal(n1, n2)  # noise-free path is bit-exact
    a = XcistAcq(kvp=120, n_view=24, quantum_noise=True, seed=5)
    r1 = run_xcist(tmp_path / "a", f, ["water"], DX, a)["prep"]
    r2 = run_xcist(tmp_path / "b", f, ["water"], DX, a)["prep"]
    r3 = run_xcist(
        tmp_path / "c", f, ["water"], DX, XcistAcq(kvp=120, n_view=24, quantum_noise=True, seed=6)
    )["prep"]
    sd = np.std(r1 - n1)
    assert sd > 0
    # measured: XCIST's quantum RNG is nearly but not exactly reproducible (1 of 21600 values
    # differed, by ~0.2 sd); bound it rather than claim bit-exactness
    assert np.mean(r1 != r2) < 1e-3 and np.abs(r1 - r2).max() < 0.5 * sd
    assert np.std(r1 - r3) > 0.5 * sd  # a different seed gives a different noise realisation


def test_recon_scaling_hu_nominal_mu(tmp_path: Path) -> None:
    pytest.importorskip("ctsynth")
    acq = XcistAcq(kvp=None, mono_keV=E0, n_view=360)
    f = fractions([Disk(0, 0, 90.0, (1.0,))], N, DX)
    try:
        out = run_xcist(tmp_path / "w", f, ["water"], DX, acq, recon=True)
    except OSError as exc:  # prebuilt macOS FDK needs libgomp (see docs/xcist.md)
        pytest.skip(f"FDK library unavailable: {exc}")
    img = out["recon"]
    yy, xx = np.mgrid[:N, :N]
    inside = (xx - (N - 1) / 2) ** 2 + (yy - (N - 1) / 2) ** 2 < (45 / DX) ** 2
    expect_hu = (MU["water"] / 0.02 - 1) * 1000  # XCIST HU uses fixed nominal mu = 0.02/mm
    assert img[inside].mean() == pytest.approx(expect_hu, abs=30.0)  # 3% of mu_water


def test_record_roundtrip_declares_missing_recon(run: dict, tmp_path: Path) -> None:
    f = fractions(_two_disks(), N, DX)
    p = to_paired_record(
        run,
        f,
        ("water", "bone"),
        np.array([1.0, 1.92]),
        ACQ,
        phantom_id="G1",
        acquisition_id="mono60",
        dest=tmp_path / "rec",
        attenuation_source="Geant4-EDLP97 (XCIST GetMu)",
    )
    rec = load_record(p)
    assert "recon" in rec.unknown and rec.acquisition["recon"] is None
    assert rec.producer["name"] == "XCIST/gecatsim"
    assert np.array_equal(np.load(p / "truth.npy"), f)  # truth is the input phantom
