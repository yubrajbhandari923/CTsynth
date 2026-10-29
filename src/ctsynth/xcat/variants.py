"""Controlled acquisition / reconstruction variants for the synthetic CT dataset.

Families: A clean baseline; B photon statistics (clean projections reused); C reconstruction kernel
(projection-domain channel
apodisation before the helical FBP, i.e. a windowed filter; not post-reconstruction smoothing); D
acquisition (energy, pitch, angular
sampling: needs new projections); E combinations drawn from documented bounded ranges with
deterministic seeds.
Variants sharing (energy, pitch, view decimation, margin) share ONE forward projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ctsynth.seeding import rng
from ctsynth.xcat.leap_sim import Acquisition

N0_LEVELS = {
    "n0_1e6": 1e6,
    "n0_3e5": 3e5,
    "n0_1e5": 1e5,
    "n0_3e4": 3e4,
}  # about 9 / 17 / 29 / 54 HU SD in muscle at 0.8 mm (measured)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    family: str
    acq: Acquisition = Acquisition()
    n0: float | None = None  # None = noise-free
    apod_sigma_px: float = 0.0  # Gaussian apodisation along detector channels (pixels) before FBP
    # LEAP set_FBPlowpass(W): 1 = off; effect on helical FBP verified empirically
    # (set_rampFilter has none)
    lowpass_w: float = 1.0

    def group_key(self) -> tuple[float, float, int, float]:
        return (self.acq.energy_kev, self.acq.pitch, self.acq.view_decimation, self.acq.margin_mm)

    def cond_vector(self) -> tuple[float, ...]:
        """Metadata for the conditioned network: [noisy, log10(n0/1e5), apod sigma, pitch,
        1/decimation, energy/70]."""
        return (
            0.0 if self.n0 is None else 1.0,
            0.0 if self.n0 is None else math.log10(self.n0 / 1e5),
            self.apod_sigma_px,
            self.lowpass_w - 1.0,
            self.acq.pitch,
            1.0 / self.acq.view_decimation,
            self.acq.energy_kev / 70.0,
        )


BASE = Acquisition()


def family_variants() -> dict[str, VariantSpec]:
    v: dict[str, VariantSpec] = {"clean": VariantSpec("clean", "A")}
    for k, n0 in N0_LEVELS.items():
        v[k] = VariantSpec(k, "B", n0=n0)
    v["apod_0p75"] = VariantSpec("apod_0p75", "C", n0=3e5, apod_sigma_px=0.75)
    v["apod_1p5"] = VariantSpec("apod_1p5", "C", n0=3e5, apod_sigma_px=1.5)
    v["apod_1p5_clean"] = VariantSpec("apod_1p5_clean", "C", apod_sigma_px=1.5)
    v["lowpass_w2"] = VariantSpec("lowpass_w2", "C", n0=3e5, lowpass_w=2.0)
    v["lowpass_w4"] = VariantSpec("lowpass_w4", "C", n0=3e5, lowpass_w=4.0)
    v["lowpass_w2_clean"] = VariantSpec("lowpass_w2_clean", "C", lowpass_w=2.0)
    v["lowpass_w4_clean"] = VariantSpec("lowpass_w4_clean", "C", lowpass_w=4.0)
    v["e60"] = VariantSpec("e60", "D", acq=Acquisition(energy_kev=60.0), n0=3e5)
    v["e80"] = VariantSpec("e80", "D", acq=Acquisition(energy_kev=80.0), n0=3e5)
    v["pitch1p4"] = VariantSpec("pitch1p4", "D", acq=Acquisition(pitch=1.4), n0=3e5)
    v["dec2"] = VariantSpec("dec2", "D", acq=Acquisition(view_decimation=2), n0=3e5)
    return v


def sample_combined(seed: int, index: int) -> VariantSpec:
    """Family E: one combination from bounded ranges (log-uniform n0 in [3e4, 1e6], sigma in {0,
    0.75, 1.5} px, energy in {60, 70, 80} keV,
    pitch in {1.0, 1.2, 1.4}, decimation in {1, 2}); deterministic in (seed, index)."""
    r = rng(seed, 5, index)
    n0 = float(10 ** r.uniform(math.log10(3e4), math.log10(1e6)))
    acq = Acquisition(
        energy_kev=float(r.choice([60.0, 70.0, 80.0])),
        pitch=float(r.choice([1.0, 1.2, 1.4])),
        view_decimation=int(r.choice([1, 2])),
    )
    return VariantSpec(
        f"combo_{index:03d}",
        "E",
        acq,
        n0,
        float(r.choice([0.0, 0.75])),
        float(r.choice([1.0, 1.0, 2.0, 4.0])),
    )


def apodise_channels(p: torch.Tensor, sigma_px: float, view_chunk: int = 500) -> torch.Tensor:
    """Gaussian filter along the detector channel axis of (view, row, channel) projections, in
    place per chunk; sigma 0 is a no-op.
    Smoothing before the (linear) derivative + Hilbert filtering of the FBP multiplies its
    frequency response by a Gaussian window."""
    if sigma_px <= 0:
        return p
    r = int(math.ceil(4 * sigma_px))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=p.device)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2)
    k = (k / k.sum())[None, None]
    for v0 in range(0, p.shape[0], view_chunk):
        c = p[v0 : v0 + view_chunk]
        n, rows, ch = c.shape
        out = F.conv1d(F.pad(c.reshape(n * rows, 1, ch), (r, r), mode="replicate"), k)
        p[v0 : v0 + view_chunk] = out.reshape(n, rows, ch)
    return p
