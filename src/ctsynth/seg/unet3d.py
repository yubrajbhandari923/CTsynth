"""Compact 3D U-Net with optional FiLM conditioning on acquisition metadata.

Input (B, in_ch, D, H, W) CT, normalised by `normalise_hu`; output logits (B, n_classes, D, H, W).
Patch sides must be divisible by 2**depth.
With cond_dim > 0 a metadata vector (B, cond_dim) scales and shifts the bottleneck and decoder
features (FiLM); with cond_dim = 0 it is a
plain U-Net and `cond` must be None.
"""

from __future__ import annotations

import torch
from torch import nn


def normalise_hu(hu: torch.Tensor) -> torch.Tensor:
    """Clip to [-1024, 1500] HU and scale to about [-1, 1.5]."""
    return hu.clamp(-1024.0, 1500.0) / 1000.0


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, cond_dim: int = 0) -> None:
        super().__init__()
        self.c1 = nn.Conv3d(c_in, c_out, 3, padding=1)
        self.n1 = nn.GroupNorm(min(8, c_out), c_out)
        self.c2 = nn.Conv3d(c_out, c_out, 3, padding=1)
        self.n2 = nn.GroupNorm(min(8, c_out), c_out)
        self.film = nn.Linear(cond_dim, 2 * c_out) if cond_dim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)  # starts as the identity modulation
            nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.relu(self.n1(self.c1(x)))
        x = self.n2(self.c2(x))
        if self.film is not None:
            if cond is None:
                raise ValueError("this network is conditioned: pass cond")
            gamma, beta = self.film(cond).chunk(2, dim=1)
            x = x * (1 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]
        return torch.relu(x)


class UNet3D(nn.Module):
    def __init__(
        self, in_ch: int = 1, n_classes: int = 15, base: int = 16, depth: int = 3, cond_dim: int = 0
    ) -> None:
        super().__init__()
        self.depth, self.cond_dim = depth, cond_dim
        ch = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList(
            [ConvBlock(in_ch if i == 0 else ch[i - 1], ch[i]) for i in range(depth)]
        )
        self.bottleneck = ConvBlock(ch[depth - 1], ch[depth], cond_dim)
        self.up = nn.ModuleList(
            [nn.ConvTranspose3d(ch[i + 1], ch[i], 2, stride=2) for i in range(depth)]
        )
        self.dec = nn.ModuleList([ConvBlock(2 * ch[i], ch[i], cond_dim) for i in range(depth)])
        self.head = nn.Conv3d(ch[0], n_classes, 1)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if (cond is None) != (self.cond_dim == 0):
            raise ValueError("cond must be given iff the network was built with cond_dim > 0")
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x, cond)
        for i in reversed(range(self.depth)):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], dim=1), cond)
        return self.head(x)  # type: ignore[no-any-return]
