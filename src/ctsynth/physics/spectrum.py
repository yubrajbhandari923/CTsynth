"""Source spectrum, filtration and detector energy response."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ctsynth.physics.materials import _all_finite, _check_common


@dataclass(frozen=True, eq=False)
class Spectrum:
    """Detected-spectrum description on an energy grid.

    Expected noiseless, scatter-free counts of detector bin b for line integrals A are
        lambda_b = sum_e response[b,e] * fluence[e] * filter_transmission[e]
                         * exp(-sum_m A_m kappa_m(E_e))
    a plain sum over grid index e (no extra quadrature weight, no dE factor).

    Attributes:
        energies: (n_energy,) strictly increasing keV, > 0.
        fluence: (n_energy,) >= 0. Photons per detector element per unit exposure that the source
            emits with energy in the bin represented by E_e (already integrated over that bin's
            width; units photons, NOT photons/keV).
        filter_transmission: (n_energy,) in [0, 1]; source/bowtie/added filtration.
        response: (n_bin, n_energy) >= 0, dimensionless weight turning one transmitted photon of
            energy E_e into counts in bin b. Energy-integrating bin: R[b, e] = E_e (or any
            weight); photon-counting bin: R[b, e] = 1 for E_e in [lo_b, hi_b), else 0.
            Rows may overlap.

    Validation (ValueError): shapes, finiteness, signs, T <= 1, strictly increasing energies,
    same dtype/device.
    """

    energies: torch.Tensor
    fluence: torch.Tensor
    filter_transmission: torch.Tensor
    response: torch.Tensor

    def __post_init__(self) -> None:
        _check_common(self.energies, self.fluence, self.filter_transmission, self.response)
        e = self.energies
        n = e.numel()
        if e.dim() != 1 or n < 1 or not _all_finite(e) or not bool((e > 0).all()):
            raise ValueError("energies must be a 1-D finite positive grid")
        if n > 1 and not bool((e[1:] > e[:-1]).all()):
            raise ValueError("energies must be strictly increasing")
        phi, trans, r = self.fluence, self.filter_transmission, self.response
        if phi.shape != (n,) or not _all_finite(phi) or not bool((phi >= 0).all()):
            raise ValueError("fluence must be (n_energy,), finite and >= 0")
        if (
            trans.shape != (n,)
            or not _all_finite(trans)
            or not bool(((trans >= 0) & (trans <= 1)).all())
        ):
            raise ValueError("filter_transmission must be (n_energy,) in [0, 1]")
        if r.dim() != 2 or r.shape[0] < 1 or r.shape[1] != n:
            raise ValueError("response must be (n_bin, n_energy) with n_bin >= 1")
        if not _all_finite(r) or not bool((r >= 0).all()):
            raise ValueError("response must be finite and >= 0")

    @property
    def n_bin(self) -> int:
        """Number of detector bins."""
        return self.response.shape[0]

    def weights(self) -> torch.Tensor:
        """W[b, e] = response[b, e] * fluence[e] * filter_transmission[e], shape (n_bin, n_energy).

        Must keep the autograd graph to `fluence` (and `filter_transmission`, `response`).
        """
        return self.response * (self.fluence * self.filter_transmission)[None, :]

    def unattenuated_counts(self) -> torch.Tensor:
        """(n_bin,) = W.sum(dim=1): expected counts with zero attenuation."""
        return self.weights().sum(dim=1)

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> Spectrum:
        """Copy with tensors moved/cast."""
        return Spectrum(
            self.energies.to(device=device, dtype=dtype),
            self.fluence.to(device=device, dtype=dtype),
            self.filter_transmission.to(device=device, dtype=dtype),
            self.response.to(device=device, dtype=dtype),
        )
