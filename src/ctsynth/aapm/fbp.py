"""Differentiable torch port of the OFFICIAL AAPM image formation (kit fanbeam_fbp_de.py).

transmission (B, 256, 1024) of one kVp -> -log -> interlaced into the 512-view sequence (80 kVp even
views, 50 kVp odd, other views empty) -> data weighting -> ramp-kernel convolution -> weighted
pixel-driven fan-beam back-projection with linear interpolation over detector bins -> x2 (only every
other view is present). Output: image[ix, iy] in cm^-1-like units (the kit's `*Images.npy`).
Linear in the filtered sinogram, so autograd is the exact adjoint-based gradient. float64 geometry.
"""

from __future__ import annotations

import math

import torch

from ctsynth.aapm.schema import KVP_HIGH, REAL_GEOMETRY, GeometrySpec


def _ramp_kernel(n: int, du: float) -> torch.Tensor:
    """Kak & Slaney ramp kernel as in the kit (odd length n = 2*nbins-1), /2, float64."""
    h = (n - 1) // 2
    nr = torch.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1.0, dtype=torch.float64)
    rf = torch.zeros(n, dtype=torch.float64)
    # kit: odd offsets from the centre = -1/(pi du nr)^2, even offsets 0, centre 1/(4 du^2)
    pos = torch.arange(h + 1, n, 2)
    neg = torch.arange(h - 1, -1, -2)
    rf[pos] = -1.0 / (math.pi * du * nr[pos]) ** 2
    rf[neg] = -1.0 / (math.pi * du * nr[neg]) ** 2
    rf[h] = 1.0 / (4.0 * du * du)
    return rf / 2.0


class AapmFBP:
    """Reusable operator (geometry/filter tensors built once per device)."""

    def __init__(
        self, geometry: GeometrySpec = REAL_GEOMETRY, device: torch.device | str = "cpu"
    ) -> None:
        g, dev = geometry, torch.device(device)
        self.g, self.dev = g, dev
        n_bins, n_views, r, sdd = g.n_det, g.n_views, g.source_radius_cm, g.sdd_cm
        length = g.detector_length_cm()
        du = length / n_bins
        self.du, self.u0, self.length = du, -length / 2.0, length
        dup = du * r / sdd
        rk = _ramp_kernel(2 * n_bins - 1, dup) * dup
        # out[j] = sum_k a[k] rk[j-k+N-1]  (numpy convolve, 'valid')
        j = torch.arange(n_bins)[:, None]
        k = torch.arange(n_bins)[None, :]
        self.toeplitz = rk[j - k + n_bins - 1].to(dev)  # (N, N) float64
        u = torch.arange(n_bins, dtype=torch.float64) * du + (-length / 2.0 + du / 2.0)
        u = u * r / sdd
        self.data_weight = (r / torch.sqrt(r**2 + u**2)).to(dev)
        n = g.n_pix
        d = g.fov_cm / n
        px = -g.fov_cm / 2.0 + d * (torch.arange(n, dtype=torch.float64) + 0.5)
        self.px, self.py = (t.to(dev) for t in torch.meshgrid(px, px, indexing="ij"))
        self.frad = torch.sqrt(self.px**2 + self.py**2)
        self.fphi = torch.atan2(self.py, self.px)
        self.inside = self.frad <= g.fov_cm / 2.0
        self.ds = 2.0 * math.pi / n_views

    def _geometry(self, view: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.g
        r, sdd = g.source_radius_cm, g.sdd_cm
        s = view * self.ds
        c, sn = math.cos(s), math.sin(s)
        xs, ys = r * c, r * sn
        xc, yc = (r - sdd) * c, (r - sdd) * sn
        bigu = (r + self.frad * torch.sin(s - self.fphi - math.pi / 2.0)) / r
        w = 1.0 / (bigu * bigu)
        ew = (self.px - xs) * c + (self.py - ys) * sn
        ratio = -sdd / ew
        dx, dy = xs + ratio * (self.px - xs), ys + ratio * (self.py - ys)
        upos = (dx - xc) * (-sn) + (dy - yc) * c
        ok = (
            self.inside
            & (upos - self.u0 >= self.du / 2)
            & (upos - self.u0 < self.length - self.du / 2)
        )
        loc = (upos - self.u0) / self.du + 0.5
        n1 = torch.floor(loc).long() - 1
        frac = loc - torch.floor(loc)
        n1 = torch.where(ok, n1, torch.zeros_like(n1))
        return w * ok, n1, frac, ok

    def __call__(self, transmission: torch.Tensor, kvp: int) -> torch.Tensor:
        """(B, V/2, n_det) air-normalised transmission of `kvp` -> (B, n_pix, n_pix) image."""
        b = transmission.shape[0]
        dtype = transmission.dtype
        sino = -torch.log(transmission.to(torch.float64))
        work = (sino * self.data_weight) @ self.toeplitz.T  # (B, V/2, N)
        start = 0 if kvp == KVP_HIGH else 1
        n = self.g.n_pix
        img = torch.zeros(b, n, n, dtype=torch.float64, device=transmission.device)
        for i in range(work.shape[1]):
            w, n1, frac, _ = self._geometry(start + 2 * i)
            v1 = work[:, i].index_select(1, n1.reshape(-1)).reshape(b, n, n)
            v2 = work[:, i].index_select(1, (n1 + 1).reshape(-1)).reshape(b, n, n)
            img = img + w * (frac * v2 + (1.0 - frac) * v1) * self.ds
        return (2.0 * img).to(dtype)
