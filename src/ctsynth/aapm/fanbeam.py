"""Differentiable torch port of the OFFICIAL AAPM fan-beam projector
(starting_kit/model/projection.py).

The kit does NOT use exact pixel chords: per ray it steps layer by layer along the dominant axis,
adding `travPixlen * pixel` for one pixel, or splitting `travPixlen` linearly between the two pixels
the ray straddles in that layer. This is linear in the image, so autograd through the gather is the
exact adjoint. Geometry/units as `schema.GeometrySpec`; output is a line integral in cm when
the image is a dimensionless volume fraction. Image axis order is `image[ix, iy]` (axis 0 = x).

Geometry is float64 to match the kit's floor() decisions; only the image dtype follows the input.
"""

from __future__ import annotations

import math

import torch

from ctsynth.aapm.schema import REAL_GEOMETRY, GeometrySpec


def _layer_sum(
    img: torch.Tensor,
    src: tuple[torch.Tensor, torch.Tensor],
    diff: tuple[torch.Tensor, torch.Tensor],
    n: tuple[int, int],
    d: tuple[float, float],
    lo: tuple[float, float],
) -> torch.Tensor:
    """Kit loop for rays whose dominant axis is `a` (index 0): img (B, na, nb), rays (R,) -> (B, R).

    src/diff/n/d/lo are ordered (layered axis a, other axis b). Weights depend on geometry only.
    """
    (sa, sb), (da, db), (na, nb), (pa, pb), (la, lb) = src, diff, n, d, lo
    slope = db / da
    trav = pa * torch.sqrt(1.0 + slope * slope)  # (R,)
    xa = la + pa * (torch.arange(na, dtype=sa.dtype, device=sa.device) + 1.0)  # far edges (na,)
    yint = sb[:, None] + slope[:, None] * (xa[None, :] - sa[:, None])  # (R, na)
    yold0 = sb + slope * (la - sa)
    yold = torch.cat([yold0[:, None], yint[:, :-1]], dim=1)
    ib = torch.floor((yint - lb) / pb).long()
    ibo = torch.floor((yold - lb) / pb).long()
    same = ib == ibo
    ymid = pb * torch.maximum(ib, ibo).to(sa.dtype) + lb
    d1, d2 = (ymid - yold).abs(), (yint - ymid).abs()
    w_old = torch.where(same, torch.ones_like(d1), d1 / (d1 + d2).clamp_min(1e-300))
    w_new = torch.where(same, torch.zeros_like(d1), 1.0 - w_old)
    out = torch.zeros(img.shape[0], ib.shape[0], dtype=img.dtype, device=img.device)
    ia = torch.arange(na, device=img.device)[None, :].expand_as(ib)
    for ibx, w in ((ibo, w_old), (ib, w_new)):
        ok = (ibx >= 0) & (ibx < nb)
        flat = (ia * nb + ibx.clamp(0, nb - 1)).reshape(-1)
        vals = img.reshape(img.shape[0], -1)[:, flat].reshape(img.shape[0], *ib.shape)
        wt = (w * ok * trav[:, None]).to(img.dtype)
        out = out + (vals * wt).sum(-1)
    return out


def fanbeam_project(
    image: torch.Tensor,
    geometry: GeometrySpec = REAL_GEOMETRY,
    views: torch.Tensor | None = None,
    ray_chunk: int = 4096,
) -> torch.Tensor:
    """(B, nx, ny) fractions -> (B, V, n_det) line integrals [cm]; differentiable in `image`.

    `views`: indices into the geometry's full `n_views` sequence (default all). Even indices are the
    80-kVp views, odd the 50-kVp views (F-032): use `arange(0, n, 2)` / `arange(1, n, 2)`.
    `ray_chunk` bounds memory (rays per pass); results do not depend on it.
    """
    g = geometry
    if image.ndim != 3 or image.shape[1:] != (g.n_pix, g.n_pix):
        raise ValueError(f"image must be (B,{g.n_pix},{g.n_pix}), got {tuple(image.shape)}")
    dev = image.device
    idx = torch.arange(g.n_views, device=dev) if views is None else views.to(dev)
    s = 2.0 * math.pi * idx.to(torch.float64) / g.n_views  # (V,)
    ulen = g.detector_length_cm()
    u = -ulen / 2 + (torch.arange(g.n_det, dtype=torch.float64, device=dev) + 0.5) * ulen / g.n_det
    r, sdd = g.source_radius_cm, g.sdd_cm
    c, sn = torch.cos(s)[:, None], torch.sin(s)[:, None]
    xs, ys = (r * c).expand(-1, g.n_det), (r * sn).expand(-1, g.n_det)
    xb = (r - sdd) * c - sn * u[None, :]
    yb = (r - sdd) * sn + c * u[None, :]
    xs, ys, dxr, dyr = (t.reshape(-1) for t in (xs, ys, xb - xs, yb - ys))
    d = g.pixel_cm()
    n, lo = g.n_pix, -g.fov_cm / 2
    xdom = dxr.abs() * d > dyr.abs() * d  # kit: xad > yad (equal pixel sizes)
    out = torch.zeros(image.shape[0], xs.numel(), dtype=image.dtype, device=dev)
    for dom in (True, False):
        sel = torch.nonzero(xdom == dom).squeeze(1)
        im = image if dom else image.transpose(1, 2)
        for k in range(0, sel.numel(), ray_chunk):
            j = sel[k : k + ray_chunk]
            a, b = (xs[j], ys[j]) if dom else (ys[j], xs[j])
            da, db = (dxr[j], dyr[j]) if dom else (dyr[j], dxr[j])
            out[:, j] = _layer_sum(im, (a, b), (da, db), (n, n), (d, d), (lo, lo))
    return out.reshape(image.shape[0], idx.numel(), g.n_det)
