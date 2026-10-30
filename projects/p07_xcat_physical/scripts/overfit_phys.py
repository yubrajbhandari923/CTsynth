#!/usr/bin/env python
"""Stage A technical check: overfit a fixed set of patches of one patient, verify gradients, checkpoint round-trip,
whole-volume prediction export (uint16) and the c -> mu70 -> HU conversion against the stored target.
usage: overfit_phys.py --cache CACHE --patient 22728 --out DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from train_phys import build_records  # noqa: E402

from ctsynth.seg.physical import PhysPatchDataset, PhysUNet, phys_loss, predict_conc
from ctsynth.xcat.cache import VolumeReader
from ctsynth.xcat.materials import ElementTable
from ctsynth.xcat.physical import conc_to_u16, load_basis, mu_mm_from_concentration, u16_to_conc

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--patient", default="22728")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=600)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda:0")
basis, et = load_basis(), ElementTable()
kap = torch.from_numpy(basis.kappa_cm2g(70.0, et)).float().to(dev)
muw = et.kappa_water(70.0)
rec = build_records(a.cache, [a.patient], ["clean"])
ds = PhysPatchDataset(rec, 96, 4, seed=0)
x, y, _ = (torch.stack(t) for t in zip(*[ds[i] for i in range(4)], strict=True))
x, y = x.to(dev), y.to(dev)
torch.manual_seed(0)
net = PhysUNet(basis.k, 16, 3).to(dev)
opt = torch.optim.AdamW(net.parameters(), 2e-3)
hist = []
for s in range(a.steps):
    opt.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = net(x)
    L = phys_loss(p.float(), y, kap, muw)
    L["total"].backward()
    if s == 0:
        gn = float(
            torch.sqrt(
                sum((q.grad.float() ** 2).sum() for q in net.parameters() if q.grad is not None)
            )
        )
    opt.step()
    if s % 50 == 0 or s == a.steps - 1:
        hist.append({"step": s, **{k: float(v) for k, v in L.items()}})
        print(hist[-1], flush=True)
with torch.no_grad():
    p = net(x).float()
res = {
    "first_loss": hist[0]["total"],
    "last_loss": hist[-1]["total"],
    "grad_norm_step0": gn,
    "min_output": float(p.min()),
    "patch_conc_mae_gcm3": float((p - y).abs().mean()),
    "patch_mu70_mae_hu": float(
        ((p - y) * kap.reshape(1, -1, 1, 1, 1)).sum(1).abs().mean() / muw * 1000
    ),
}
torch.save({"net": net.state_dict()}, out / "ck.pt")
net2 = PhysUNet(basis.k, 16, 3).to(dev)
net2.load_state_dict(torch.load(out / "ck.pt")["net"])
with torch.no_grad():
    res["checkpoint_roundtrip_max_diff"] = float((net2(x).float() - p).abs().max())
hu = VolumeReader(rec[0].ct_path, "npy_i16q").read_all()
pred = predict_conc(net, hu, 96, 64, 4, None, dev)
np.save(out / "pred_conc.npy", conc_to_u16(pred.cpu().numpy()))
back = u16_to_conc(np.load(out / "pred_conc.npy"))
res["export_roundtrip_max_abs"] = float(np.abs(back - pred.cpu().numpy()).max())
res["export_shape"] = list(back.shape)
mu_mm = mu_mm_from_concentration(torch.from_numpy(back).to(dev), kap)
res["export_mu70_hu_range"] = [
    float(1000 * (mu_mm.min() / (muw / 10) - 1)),
    float(1000 * (mu_mm.max() / (muw / 10) - 1)),
]
(out / "overfit.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
