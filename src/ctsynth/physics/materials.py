"""Material basis: mass-attenuation tables on an energy grid. Units: keV, g/cm^3, cm^2/g."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _check_common(*tensors: torch.Tensor) -> None:
    """ValueError unless all tensors are floating point with one shared dtype and device."""
    first = tensors[0]
    for t in tensors:
        if not t.is_floating_point():
            raise ValueError(f"tensor must be floating point, got {t.dtype}")
        if t.dtype != first.dtype or t.device != first.device:
            raise ValueError("tensors must share dtype and device")


def _all_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


@dataclass(frozen=True, eq=False)
class MaterialBasis:
    """Basis of `n_material` materials.

    Attributes:
        names: unique material names, length n_material.
        ref_density: (n_material,) reference (pure-material) density rho_ref_m in g/cm^3, > 0.
        energies: (n_energy,) strictly increasing grid in keV, > 0, n_energy >= 2.
        kappa: (n_material, n_energy) mass attenuation mu/rho in cm^2/g, finite and > 0.

    Validation (in `__post_init__`, raises ValueError): shapes, finiteness, positivity,
    strictly increasing energies, unique names, floating dtype, all tensors same dtype/device.
    """

    names: tuple[str, ...]
    ref_density: torch.Tensor
    energies: torch.Tensor
    kappa: torch.Tensor

    def __post_init__(self) -> None:
        _check_common(self.ref_density, self.energies, self.kappa)
        n_mat = len(self.names)
        if n_mat == 0 or len(set(self.names)) != n_mat:
            raise ValueError("names must be non-empty and unique")
        rd, e, k = self.ref_density, self.energies, self.kappa
        if rd.shape != (n_mat,) or not _all_finite(rd) or not bool((rd > 0).all()):
            raise ValueError("ref_density must be (n_material,), finite and > 0")
        if e.dim() != 1 or e.numel() < 2 or not _all_finite(e) or not bool((e > 0).all()):
            raise ValueError("energies must be a 1-D finite grid of at least 2 positive values")
        if not bool((e[1:] > e[:-1]).all()):
            raise ValueError("energies must be strictly increasing")
        if k.shape != (n_mat, e.numel()) or not _all_finite(k) or not bool((k > 0).all()):
            raise ValueError("kappa must be (n_material, n_energy), finite and > 0")

    @property
    def n_material(self) -> int:
        """Number of materials."""
        return len(self.names)

    def kappa_at(self, energies: torch.Tensor) -> torch.Tensor:
        """kappa on arbitrary energies (keV), shape (n_material, len(energies)).

        Log-log linear interpolation: ln kappa is piecewise linear in ln E between grid nodes.
        Energies must lie in [energies[0], energies[-1]] (closed interval); otherwise ValueError
        (no extrapolation). Differentiable w.r.t. `kappa`.
        """
        grid = self.energies
        if energies.numel() and (
            bool((energies < grid[0]).any()) or bool((energies > grid[-1]).any())
        ):
            raise ValueError(
                f"energies outside basis grid [{float(grid[0])}, {float(grid[-1])}] keV"
            )
        lg, le = torch.log(grid), torch.log(energies)
        n = grid.numel()
        idx = (torch.searchsorted(lg, le, right=True) - 1).clamp(0, n - 2)
        t = (le - lg[idx]) / (lg[idx + 1] - lg[idx])
        lnk = torch.log(self.kappa)
        return torch.exp(lnk[:, idx] * (1 - t) + lnk[:, idx + 1] * t)

    def mass_density_from_volume_fraction(self, fraction: torch.Tensor) -> torch.Tensor:
        """rho_m = f_m * rho_ref_m; `fraction` (n_material, ...) -> same shape, g/cm^3."""
        if fraction.dim() < 1 or fraction.shape[0] != self.n_material:
            raise ValueError("fraction must have leading dimension n_material")
        ref = self.ref_density.reshape((-1,) + (1,) * (fraction.dim() - 1))
        return fraction * ref

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> MaterialBasis:
        """Copy with tensors moved/cast."""
        return MaterialBasis(
            self.names,
            self.ref_density.to(device=device, dtype=dtype),
            self.energies.to(device=device, dtype=dtype),
            self.kappa.to(device=device, dtype=dtype),
        )
