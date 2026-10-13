"""Tests for the P04 XCIST drivers that need no XCIST install (the real simulator is covered by
the integration tests in tests/paired/test_xcist.py).

What these protect: dataset-definition invariants, the ray-march / analytic-chord agreement that
the mismatch ladder (EXP-010) depends on (including orientation), and the file contract between the
HPC replication job and the laptop comparison (RUN-0022 / F-036). They do not test XCIST physics.
"""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ctsynth.hashing import sha256_file
from ctsynth.paired.phantoms import Disk, chords, fractions, voxel_chords
from ctsynth.paired.xcist import XcistAcq, fan_rays

P04 = Path(__file__).resolve().parents[2] / "projects" / "p04_xcist"
sys.path.insert(0, str(P04))
generate = importlib.import_module("generate")
compare = importlib.import_module("compare")
compare_replication = importlib.import_module("compare_replication")
hpc_replicate = importlib.import_module("hpc_replicate")


# --- dataset definition (generate.py) ------------------------------------------------------------
@pytest.mark.parametrize("pid", sorted(generate.PHANTOMS))
def test_phantoms_are_valid_volume_fraction_truth(pid: str) -> None:
    f = fractions(generate.PHANTOMS[pid], generate.N, generate.DX)
    assert f.shape == (3, 1, generate.N, generate.N)
    assert f.min() > -1e-9 and f.max() < 1 + 1e-9  # each material is a fraction
    assert f.sum(axis=0).max() < 1 + 1e-9  # water + Ca + I never exceeds a full voxel
    assert np.array_equal(f, fractions(generate.PHANTOMS[pid], generate.N, generate.DX))


@pytest.mark.parametrize("pid", sorted(generate.PHANTOMS))
def test_every_disk_lies_inside_the_field_of_view(pid: str) -> None:
    half = generate.N * generate.DX / 2
    for d in generate.PHANTOMS[pid]:
        assert max(abs(d.cx_mm), abs(d.cy_mm)) + d.r_mm <= half


def test_inserts_conserve_material_and_variants_are_declared() -> None:
    for disks in generate.PHANTOMS.values():
        for d in disks[1:]:  # inserts trade one material for another: delta sums to zero
            assert abs(sum(d.delta)) < 1e-12
    assert set(generate.VARIANTS) == {"ref", "sub", "noisy"}


# --- ray march vs exact chords: also pins the axis/orientation convention ------------------
@pytest.fixture(scope="module")
def setup() -> tuple[list[Disk], np.ndarray, np.ndarray, np.ndarray]:
    disks = [
        Disk(30.0, 20.0, 40.0, (1.0, 0.0, 0.0))
    ]  # off-centre in x AND y: no symmetry to hide behind
    src, det = fan_rays(XcistAcq(n_view=36))
    return disks, fractions(disks, 128, 2.0), src, det


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2)))


def test_voxel_chords_match_analytic_chords(setup) -> None:  # type: ignore[no-untyped-def]
    disks, f, src, det = setup
    # measured 1.4% rms (2 mm rasterisation); 3% leaves headroom without hiding an axis bug
    assert _rel_rms(voxel_chords(f, 2.0, src, det)[0], chords(disks, src, det)[0]) < 0.03


@pytest.mark.parametrize("name", ["flip_y", "flip_x", "transpose"])
def test_voxel_chords_detects_wrong_axis_order(setup, name: str) -> None:  # type: ignore[no-untyped-def]
    disks, f, src, det = setup
    bad = {"flip_y": f[:, :, ::-1], "flip_x": f[:, :, :, ::-1], "transpose": f.swapaxes(2, 3)}[name]
    err = _rel_rms(
        voxel_chords(np.ascontiguousarray(bad), 2.0, src, det)[0], chords(disks, src, det)[0]
    )
    assert err > 0.3  # measured ~1.0: a misoriented phantom is nowhere near the analytic answer


def test_metrics_on_known_values() -> None:
    ref = np.array([[0.0, 1.0, 2.0, 9.0]])  # only 1 and 2 are "through the object" (0.05, 8)
    delta = np.array([[5.0, 0.1, -0.1, 5.0]])
    m = compare.metrics(delta, ref)
    assert m["rms_obj"] == pytest.approx(0.1) and m["mean_obj"] == pytest.approx(0.0)
    assert m["rel_rms_obj"] == pytest.approx(0.1 / np.sqrt(2.5))
    assert m["max_abs"] == 5.0


def test_compare_finds_committed_tables_from_any_cwd(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    grid, kappa, spectra = compare.tables()
    assert kappa.shape[0] == 3 and len(grid) == kappa.shape[1] and "fluence_120" in spectra


# --- HPC replication contract (hpc_replicate.py -> compare_replication.py) ------------------------
def _fake_xcist(workdir, f, xn, dx, acq, recon):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    return {
        "prep": rng.random((4, 6)).astype(np.float32),
        "recon": rng.random((5, 5)).astype(np.float32),
        "provenance": {"gecatsim": "0.0-test", "numpy": np.__version__},
    }


def test_hpc_replicate_writes_registered_outputs(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(hpc_replicate, "run_xcist", _fake_xcist)
    hpc_replicate.main()
    meta = json.loads((tmp_path / "replication.json").read_text())
    assert meta["prep_sha256"] == sha256_file(tmp_path / "prep.npy")
    assert meta["recon_sha256"] == sha256_file(tmp_path / "recon.npy")
    names = [
        json.loads(line)["name"] for line in (tmp_path / "artifacts.jsonl").read_text().splitlines()
    ]
    assert names == ["replication.json", "prep.npy", "recon.npy"]
    metric = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert metric["name"] == "xcist_runtime_s" and metric["value"] >= 0


def _replication_fixture(tmp_path: Path, hpc_prep: np.ndarray) -> tuple[Path, Path, np.ndarray]:
    truth = fractions(generate.PHANTOMS["G1"], generate.N, generate.DX)
    prep, recon = np.arange(12.0).reshape(3, 4), np.arange(9.0).reshape(3, 3)
    rec = tmp_path / "ds" / "G1__kvp120_ref"
    rec.mkdir(parents=True)
    np.save(rec / "prep.npy", prep)
    np.save(rec / "recon.npy", recon)
    np.save(rec / "truth.npy", truth)
    out = tmp_path / "run" / "collected" / "outputs"
    out.mkdir(parents=True)
    np.save(out / "prep.npy", hpc_prep)
    np.save(out / "recon.npy", recon)
    meta = {
        "truth_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
        "runtime_s": 1.0,
        "host": "h",
        "provenance": {"gecatsim": "g", "numpy": "n"},
    }
    (out / "replication.json").write_text(json.dumps(meta))
    return tmp_path / "run", tmp_path / "ds", prep


def test_compare_replication_identical_and_perturbed(tmp_path: Path) -> None:
    prep = np.arange(12.0).reshape(3, 4)
    run, ds, _ = _replication_fixture(tmp_path / "same", prep)
    res = compare_replication.main(run, ds)
    assert (
        res["prep_bit_identical"] and res["truth_sha256_equal"] and res["prep_max_abs_diff"] == 0.0
    )
    assert (run / "replication_comparison.json").is_file()
    run, ds, _ = _replication_fixture(tmp_path / "diff", prep + 1e-6)
    res = compare_replication.main(run, ds)
    assert not res["prep_bit_identical"] and res["prep_max_abs_diff"] == pytest.approx(1e-6)
