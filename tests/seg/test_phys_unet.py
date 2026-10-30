"""Physical U-Net: nonnegative output, zero loss for a perfect prediction, gradients reach the
network, CT/concentration patches stay aligned (asymmetric volume), overfit on one patch, windowed
inference reproduces a per-window forward pass."""

import numpy as np
import torch

from ctsynth.seg.dataset import class_voxel_samples
from ctsynth.seg.physical import PhysPatchDataset, PhysRecord, PhysUNet, phys_loss, predict_conc
from ctsynth.xcat.cache import write_hu
from ctsynth.xcat.physical import conc_to_u16

KAP = torch.tensor([0.175, 0.19, 0.257])


def _record(tmp_path):
    lab = np.zeros((40, 48, 56), np.uint8)
    lab[10:20, 12:30, 30:50] = 1
    lab[25:35, 5:15, 5:15] = 2
    conc = np.zeros((3, *lab.shape), np.float32)
    conc[0] = 0.0012
    conc[0][lab > 0] = 0.0
    conc[1][lab == 1] = 1.0
    conc[2][lab == 2] = 1.4
    hu = lab.astype(np.float32) * 300.0 - 1000.0  # HU encodes the label
    w = write_hu(tmp_path / "ct", hu, "npy_i16q")
    np.save(tmp_path / "conc.npy", conc_to_u16(conc))
    return (
        PhysRecord(
            "p", w.path, "npy_i16q", str(tmp_path / "conc.npy"), (), class_voxel_samples(lab, 300)
        ),
        lab,
        hu,
        conc,
    )


def test_output_nonnegative_and_loss_zero_for_perfect_prediction():
    net = PhysUNet(3, base=4, depth=2)
    out = net(torch.randn(2, 1, 16, 16, 16) * 3)
    assert out.shape == (2, 3, 16, 16, 16) and float(out.min()) >= 0.0
    t = torch.rand(2, 3, 8, 8, 8)
    z = phys_loss(t, t, KAP, 0.19)
    assert all(float(v) == 0.0 for v in z.values())
    assert float(phys_loss(t + 0.1, t, KAP, 0.19)["total"]) > 0


def test_patch_alignment_ct_vs_concentration(tmp_path):
    rec, lab, _, _ = _record(tmp_path)
    ds = PhysPatchDataset([rec], 16, 20, seed=1)
    for i in range(20):
        x, y, _ = ds[i]
        hu = x[0].numpy() * 1000.0  # normalise_hu: /1000
        soft, bone = y[1].numpy() > 0.5, y[2].numpy() > 0.5
        assert y.shape == (3, 16, 16, 16)
        assert np.allclose(hu[soft], -700.0, atol=1) and np.allclose(hu[bone], -400.0, atol=1)
    assert np.array_equal(ds[3][1].numpy(), PhysPatchDataset([rec], 16, 20, seed=1)[3][1].numpy())


def test_overfits_one_patch_and_windowed_inference_matches(tmp_path):
    rec, lab, hu, conc = _record(tmp_path)
    ds = PhysPatchDataset([rec], 16, 4, seed=0)
    x, y, _ = ds[0]
    x, y = x[None], y[None]
    torch.manual_seed(0)
    net = PhysUNet(3, base=8, depth=2)
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    first = None
    for _ in range(120):
        opt.zero_grad()
        loss = phys_loss(net(x), y, KAP, 0.19)["total"]
        first = first if first is not None else float(loss)
        loss.backward()
        opt.step()
    assert float(loss) < 0.35 * first and all(p.grad is not None for p in net.parameters())
    full = predict_conc(
        net, hu[:32, :32, :32].copy(), patch=16, stride=8, batch=2, device="cpu", amp=False
    )
    assert full.shape == (3, 32, 32, 32) and float(full.min()) >= 0
    one = predict_conc(
        net, hu[:16, :16, :16].copy(), patch=16, stride=16, batch=1, device="cpu", amp=False
    )
    net.eval()
    with torch.no_grad():
        ref = net(torch.from_numpy(np.clip(hu[:16, :16, :16], -1024, 1500) / 1000.0)[None, None])[0]
    assert torch.allclose(one, ref, atol=1e-5)
