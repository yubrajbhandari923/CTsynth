"""Regression tests for facts verified against the OFFICIAL AAPM kit during the consolidation
audit (F-030, F-031, F-032, and the B.11 leakage guard).

These lock in conclusions that were checked by hand against the real kit source and real kit data.
They must not be weakened: each one caught, or would have caught, a real error.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from ctsynth.aapm import classical as C
from ctsynth.aapm import metrics as M
from ctsynth.aapm import schema as S
from ctsynth.aapm.train import train_label_free, train_supervised


def test_s1_matches_official_kit_formula() -> None:
    """F-031: our s1 equals the kit's, including its (nmat, nim, nx, ny) axis order.

    Official (starting_kit/metrics.py): per case, sum over materials and pixels of
    diff^2 / (nmat*nx*ny), square-rooted, then averaged over cases. Our arrays are
    (nim, nmat, nx, ny), so a transposed reference catches any axis-order mistake.
    rtol 1e-12: both are float64 sums of the same terms, so only reassociation differs.
    """
    rng = np.random.default_rng(0)
    nim, nmat, nx, ny = 4, 3, 16, 16
    truth, pred = rng.random((nim, nmat, nx, ny)), rng.random((nim, nmat, nx, ny))
    i, p = np.transpose(truth, (1, 0, 2, 3)), np.transpose(pred, (1, 0, 2, 3))
    official = np.sqrt((((i - p) ** 2 / float(nmat * nx * ny)).sum(axis=3).sum(axis=2)).sum(axis=0))
    assert M.s1_mean_rmse(truth, pred) == pytest.approx(official.mean(), rel=1e-12)


def test_kvp_parity_matches_official_generator() -> None:
    """F-032: kit dualEnergyTransmission.py uses Sinogram[:, ::2] for 80 kVp and [:, 1::2] for 50.

    i.e. "First view is 80 kVp, second is 50 kVp". Exact integer equality; no tolerance needed.
    """
    assert S.kvp_of_view(0) == 80 and S.kvp_of_view(2) == 80
    assert S.kvp_of_view(1) == 50 and S.kvp_of_view(3) == 50
    assert np.array_equal(S.view_indices(80, 8), np.arange(0, 8, 2))
    assert np.array_equal(S.view_indices(50, 8), np.arange(1, 8, 2))


def test_simplex_adds_nothing_beyond_clipping() -> None:
    """F-030: on data where sum_to_one leaves pixels outside [0, 1], simplex projection is
    indistinguishable from a naive clip, so a 'simplex wins' result is a clipping artifact.

    Built on synthetic measurements perturbed off the linear model (mimicking the 6-8% real-data
    mismatch of F-025). rtol 5e-2: the claim is "no materially different behaviour", not identity.
    """
    rng = np.random.default_rng(1)
    tables = {
        k: S.load_model_table(f"data/aapm_kit_sample/model_data_{k}kVp.npy", k) for k in S.KVPS
    }
    mu = C.effective_mu(tables)
    truth = rng.dirichlet((1.0, 1.0, 0.05), size=(6, 6)).transpose(2, 0, 1)[None]
    lo = (mu[0][:, None, None] * truth).sum(axis=1) * 1.07  # model mismatch, as on real data
    hi = (mu[1][:, None, None] * truth).sum(axis=1) * 0.94
    s2o = C.decompose_lstsq(lo, hi, mu, mode="sum_to_one", air_cutoff=None)
    simp = C.decompose_lstsq(lo, hi, mu, mode="simplex", air_cutoff=None)
    assert np.mean((s2o < 0) | (s2o > 1)) > 0.1, "fixture must produce out-of-range pixels"
    rmse = lambda p: float(np.sqrt(((p - truth) ** 2).mean()))  # noqa: E731
    assert rmse(simp) == pytest.approx(rmse(np.clip(s2o, 0.0, 1.0)), rel=5e-2)


def test_label_free_training_cannot_receive_ground_truth() -> None:
    """B.11: the no-leakage guarantee is structural, not documentary.

    train_label_free must take no truth-like parameter at all; train_supervised must take them
    explicitly, so label use is visible at the call site.
    """
    free = set(inspect.signature(train_label_free).parameters)
    assert not {p for p in free if "truth" in p or "label" in p or "target" in p}
    sup = set(inspect.signature(train_supervised).parameters)
    assert {"truth_train", "truth_val"} <= sup
