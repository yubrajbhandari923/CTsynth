"""T9 (finite differences) and T10 (torch gradcheck)."""

import numpy as np
import physics_reference as ref
import torch

from ctsynth.physics import SpectralForwardModel, Spectrum

F64 = torch.float64


def _small(n_view: int = 3) -> SpectralForwardModel:
    p = ref.build_projector(4, 3, 0.25, (0.0, 0.7, 1.9)[:n_view], 4, 0.3, 0.013)
    return SpectralForwardModel(ref.build_basis(), ref.build_spectrum(), p)


def test_t09_finite_difference_gradient_wrt_rho(gen: torch.Generator) -> None:
    """d/d rho of L = sum(w * counts) (w random seeded) by autograd vs central differences
    (L(x+h)-L(x-h))/2h, h=1e-6 on every pixel, float64. Truncation ~h^2 f'''/6 ~1e-12*|g| (kappa*dx
    <~ 2); roundoff ~ eps*|L|/h ~ 1e-16*1e6/1e-6 = 1e-4 absolute vs |g| ~ 1e4. So rtol 1e-6 with
    atol 1e-6*max|g| is conservative (~100x margin)."""
    m = _small()
    rho = (0.05 + 0.4 * torch.rand(2, 3, 4, generator=gen, dtype=F64)).requires_grad_(True)
    w = torch.rand(3, 3, 4, generator=gen, dtype=F64)
    loss = lambda r: (w * m.expected_counts(r)).sum()  # noqa: E731
    g = torch.autograd.grad(loss(rho), rho)[0].numpy()
    fd = np.zeros_like(g)
    h = 1e-6
    base = rho.detach()
    for idx in np.ndindex(*g.shape):
        d = torch.zeros_like(base)
        d[idx] = h
        fd[idx] = float(loss(base + d) - loss(base - d)) / (2 * h)
    assert np.allclose(g, fd, rtol=1e-6, atol=1e-6 * np.abs(g).max())


def test_t09_gradient_wrt_fluence(gen: torch.Generator) -> None:
    """Counts are linear in fluence: d counts_b / d Phi_e = R[b,e] T[e] exp(-A.kappa_e) + 0
    (reference: numpy closed form, rtol 1e-11), summed with random weights per ray via
    line integrals."""
    a = ref.toy_arrays()
    spec0 = ref.build_spectrum()
    phi = spec0.fluence.clone().requires_grad_(True)
    spec = Spectrum(spec0.energies, phi, spec0.filter_transmission, spec0.response)
    model = SpectralForwardModel(
        ref.build_basis(), spec, ref.build_projector(4, 3, 0.25, (0.0,), 2, 0.25)
    )
    at = torch.tensor([[[0.3, 0.6]], [[2.0, 4.0]]], dtype=F64)  # (2,1,2)
    wts = torch.rand(3, 1, 2, generator=gen, dtype=F64)
    (wts * model.expected_counts_from_line_integrals(at)).sum().backward()
    want = np.zeros(12)
    for r in range(2):
        att = np.exp(-(at[:, 0, r].numpy() @ a["kappa_e"]))
        for b in range(3):
            want += wts[b, 0, r].item() * a["resp"][b] * a["trans"] * att
    assert phi.grad is not None
    assert np.allclose(phi.grad.numpy(), want, rtol=1e-11)


def test_t10_gradcheck_model_and_projector(gen: torch.Generator) -> None:
    """torch.autograd.gradcheck (analytical vs numerical Jacobian) in float64: eps=1e-6,
    atol=1e-6, rtol=1e-4 (gradcheck defaults are looser). Outputs scaled to O(1)
    (counts / unattenuated total) so the absolute tolerance is meaningful. Covers
    expected_counts(rho), projector forward and adjoint, line_integrals."""
    p = ref.build_projector(4, 3, 0.25, (0.0, 0.7), 4, 0.3, 0.013)
    m = SpectralForwardModel(ref.build_basis(), ref.build_spectrum(), p)
    scale = float(ref.toy_arrays()["w"].sum())
    rho = (0.05 + 0.3 * torch.rand(2, 3, 4, generator=gen, dtype=F64)).requires_grad_(True)
    kw = {"eps": 1e-6, "atol": 1e-6, "rtol": 1e-4}
    assert torch.autograd.gradcheck(lambda r: m.expected_counts(r) / scale, (rho,), **kw)
    assert torch.autograd.gradcheck(m.line_integrals, (rho,), **kw)
    y = torch.randn(2, 2, 4, generator=gen, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(p.forward, (rho,), **kw)
    assert torch.autograd.gradcheck(p.adjoint, (y,), **kw)
