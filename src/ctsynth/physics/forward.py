"""Spectral forward model: material densities -> line integrals -> expected / sampled counts."""

from __future__ import annotations

import torch

from ctsynth.physics.materials import MaterialBasis, _all_finite
from ctsynth.physics.projector import ParallelBeamProjector2D
from ctsynth.physics.spectrum import Spectrum


class SpectralForwardModel:
    """rho (n_material, ny, nx) g/cm^3 -> counts (n_bin, n_view, n_det).

    Constructor validates: spectrum.energies inside basis grid (ValueError, via kappa_at);
    basis/spectrum/projector share dtype and device (ValueError); `scatter` is None, a float
    >= 0 (scatter_b = scatter * unattenuated_counts[b], uniform over views/detectors), or a tensor
    (n_bin, n_view, n_det) >= 0 added verbatim to expected counts (ValueError otherwise).
    """

    def __init__(
        self,
        basis: MaterialBasis,
        spectrum: Spectrum,
        projector: ParallelBeamProjector2D,
        scatter: float | torch.Tensor | None = None,
    ) -> None:
        dtype, device = basis.kappa.dtype, basis.kappa.device
        for t in (basis.energies, basis.ref_density, spectrum.energies, projector.angles):
            if t.dtype != dtype or t.device != device:
                raise ValueError("basis, spectrum and projector must share dtype and device")
        basis.kappa_at(spectrum.energies)  # raises ValueError outside the basis grid
        n_bin = spectrum.n_bin
        if isinstance(scatter, torch.Tensor):
            shape = (n_bin, projector.n_view, projector.n_det)
            if scatter.shape != shape or scatter.dtype != dtype or scatter.device != device:
                raise ValueError(f"scatter tensor must be {shape} with the model dtype/device")
            if not _all_finite(scatter) or not bool((scatter >= 0).all()):
                raise ValueError("scatter tensor must be finite and >= 0")
        elif scatter is not None:
            scatter = float(scatter)
            if not scatter >= 0:
                raise ValueError("scatter fraction must be >= 0")
        self.basis = basis
        self.spectrum = spectrum
        self.projector = projector
        self.scatter = scatter

    def _check_rho(self, rho: torch.Tensor) -> None:
        dtype, device = self.basis.kappa.dtype, self.basis.kappa.device
        if rho.dtype != dtype or rho.device != device:
            raise ValueError("rho must match the model dtype and device")

    def line_integrals(self, rho: torch.Tensor) -> torch.Tensor:
        """A_m = P rho_m, shape (n_material, n_view, n_det), g/cm^2. rho dtype must match model."""
        self._check_rho(rho)
        if rho.dim() < 2 or rho.shape[0] != self.basis.n_material:
            raise ValueError("rho must be (n_material, ny, nx)")
        return self.projector.forward(rho)

    def expected_counts_from_line_integrals(self, line_integrals: torch.Tensor) -> torch.Tensor:
        """(n_material, *S) g/cm^2 -> (n_bin, *S): lambda_b + scatter_b. Noiseless.

        Result is >= 0, finite for finite input (underflows to exactly 0 plus scatter), with finite
        gradients everywhere. A tensor `scatter` requires S == (n_view, n_det), else ValueError.
        """
        self._check_rho(line_integrals)
        if line_integrals.dim() < 1 or line_integrals.shape[0] != self.basis.n_material:
            raise ValueError("line integrals must have leading dimension n_material")
        sample_shape = tuple(line_integrals.shape[1:])
        kappa = self.basis.kappa_at(self.spectrum.energies)  # (n_material, n_energy)
        expo = torch.tensordot(kappa, line_integrals, dims=([0], [0]))  # (n_energy, *S)
        counts: torch.Tensor = torch.tensordot(
            self.spectrum.weights(), torch.exp(-expo), dims=([1], [0])
        )
        if isinstance(self.scatter, torch.Tensor):
            if sample_shape != (self.projector.n_view, self.projector.n_det):
                raise ValueError("tensor scatter requires sample shape (n_view, n_det)")
            return counts + self.scatter
        if self.scatter is not None:
            unatt = self.spectrum.unattenuated_counts()
            scatter_b: torch.Tensor = self.scatter * unatt.reshape((-1,) + (1,) * len(sample_shape))
            return counts + scatter_b
        return counts

    def expected_counts(self, rho: torch.Tensor) -> torch.Tensor:
        """Deterministic noiseless counts (n_bin, n_view, n_det); differentiable in rho,
        spectrum tensors, scatter. Uses no RNG."""
        return self.expected_counts_from_line_integrals(self.line_integrals(rho))

    def sample_counts(self, rho: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Poisson(expected_counts(rho)) via torch.poisson(rate, generator=generator).

        Only `generator` is consumed (global RNG untouched). Result detached, same dtype,
        non-negative integer-valued. Rate 0 yields 0.
        """
        with torch.no_grad():
            rate = self.expected_counts(rho)
            return torch.poisson(rate, generator=generator)

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> SpectralForwardModel:
        """New model with all components moved/cast."""
        scatter = self.scatter
        if isinstance(scatter, torch.Tensor):
            scatter = scatter.to(device=device, dtype=dtype)
        return SpectralForwardModel(
            self.basis.to(device, dtype),
            self.spectrum.to(device, dtype),
            self.projector.to(device, dtype),
            scatter,
        )
