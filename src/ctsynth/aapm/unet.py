"""Small 2D U-Net, 2 input channels (low/high attenuation) -> 3 material fractions.

The OUTPUT PARAMETRISATION is a scientific choice (F-019), selected explicitly via `output=`:

* "linear": raw, unconstrained.
* "softplus": nonnegative only. Nonnegativity alone does NOT remove the 3-from-2 ambiguity.
* "softmax": nonnegative AND sum exactly 1 at EVERY pixel. This silently imposes the identifying
  constraint everywhere, which is WRONG for air pixels (~35% of the kit) and band pixels (~5%).
* "softmax_with_air": 4-way softmax over [adipose, fibroglandular, calcification, air], the air
  channel dropped, so fractions are >= 0 and sum to <= 1 (a true air pixel is representable).

Input height/width must be divisible by 2**depth.
"""

from __future__ import annotations

import torch
from torch import nn

OUTPUTS = ("linear", "softplus", "softmax", "softmax_with_air")


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self, depth: int = 2, width: int = 8, output: str = "linear") -> None:
        super().__init__()
        if output not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS}, got {output!r}")
        self.output, self.depth = output, depth
        ws = [width * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList(
            [_block(2 if i == 0 else ws[i - 1], ws[i]) for i in range(depth + 1)]
        )
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(ws[i + 1], ws[i], 2, stride=2) for i in range(depth)]
        )
        self.dec = nn.ModuleList([_block(2 * ws[i], ws[i]) for i in range(depth)])
        self.head = nn.Conv2d(width, 4 if output == "softmax_with_air" else 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2**self.depth or x.shape[-2] % 2**self.depth:
            raise ValueError(f"H and W must be divisible by {2**self.depth}")
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else nn.functional.max_pool2d(x, 2))
            skips.append(x)
        for i in reversed(range(self.depth)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1))
        z = self.head(x)
        if self.output == "softplus":
            return nn.functional.softplus(z)
        if self.output == "softmax":
            return torch.softmax(z, dim=1)
        if self.output == "softmax_with_air":
            return torch.softmax(z, dim=1)[:, :3]
        return z  # type: ignore[no-any-return]
