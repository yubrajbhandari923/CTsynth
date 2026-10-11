"""Fast tests for EXP-002 estimator/runner (tiny problem sizes)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

EXP = Path(__file__).resolve().parents[2] / "projects/p01_material_inference/exp002"
sys.path.insert(0, str(EXP))
import run_exp002 as run  # noqa: E402
from estimator import DT, build_design, counts, fisher_analysis, lm_poisson  # noqa: E402

from ctsynth.config import load_config  # noqa: E402

ROOT = EXP.parents[2]


@pytest.fixture(scope="module")
def ctx():
    return run.Ctx(load_config(ROOT / "configs/exp002.yaml"))


def test_config_loads():
    cfg = load_config(ROOT / "configs/exp002.yaml")
    assert cfg.experiment_id == "EXP-002"


@pytest.mark.parametrize("name", ["M1", "M2", "M4"])
@pytest.mark.parametrize("pert", [(0.0, 0.0), (0.1, 0.0), (0.0, -0.02)])
def test_equal_total_photons(ctx, name, pert):
    d = build_design(name, ctx.spectra, ctx.E, 1e5, *pert)
    assert abs(float(d.W.sum()) - 1e5) / 1e5 < 1e-9


def test_counts_match_forward_model(ctx):
    d = build_design("M4", ctx.spectra, ctx.E, 1e6)
    A = ctx.ph["P3"]["A"].reshape(3, -1)[:, :50]
    ref = ctx.data_lambda(d.W, A)
    mine = counts(A.T.contiguous(), d.W, ctx.kappa).T
    torch.testing.assert_close(mine, ref, rtol=1e-12, atol=0)


def test_analytic_jacobian_matches_autograd(ctx):
    from estimator import jacobian

    d = build_design("M2", ctx.spectra, ctx.E, 1e6)
    A = torch.tensor([[5.0, 0.01, 0.05]], dtype=DT, requires_grad=True)
    _, J = jacobian(A, d.W, ctx.kappa)
    for b in range(d.n_meas):
        (g,) = torch.autograd.grad(counts(A, d.W, ctx.kappa)[0, b], A)
        torch.testing.assert_close(J[0, b], g[0], rtol=1e-10, atol=0)


def test_rank_predictions(ctx):
    A = torch.tensor([[8.0, 0.02, 0.1]], dtype=DT)
    rk = {}
    for name, k in (("M1", 2), ("M2", 2), ("M2", 3), ("M4", 3)):
        d = build_design(name, ctx.spectra, ctx.E, 1e6)
        rk[(name, k)] = int(fisher_analysis(A[:, :k], d.W, ctx.kappa[:k])["rank"][0])
    assert rk == {("M1", 2): 1, ("M2", 2): 2, ("M2", 3): 2, ("M4", 3): 3}


def test_lm_recovers_noiseless_truth(ctx):
    d = build_design("M2", ctx.spectra, ctx.E, 1e6)
    A = torch.tensor([[6.0, 0.02], [12.0, 0.0], [3.0, 0.05]], dtype=DT)
    y = counts(A, d.W, ctx.kappa[:2])  # noiseless, non-integer: MLE is exactly A
    Ah, conv = lm_poisson(y, d.W, ctx.kappa[:2])
    assert bool(conv.all())
    torch.testing.assert_close(Ah, A, rtol=1e-6, atol=1e-8)


def test_determinism_tiny(tmp_path):
    cfg = load_config(ROOT / "configs/exp002.yaml")
    p = {**cfg.params, "n_real": 3, "rays_per_insert": 1, "n_random_rays": 4, "full_image": False}
    tiny = cfg.model_copy(update={"params": p})
    a = json.dumps(run.run(tiny, tmp_path), sort_keys=True)
    b = json.dumps(run.run(tiny, tmp_path), sort_keys=True)
    assert a == b
    assert np.isfinite(json.loads(a)["metrics"]["jac_cond"]["P2|M2|k2|N1e+06|matched"]["median"])
