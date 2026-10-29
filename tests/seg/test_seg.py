"""U-Net shapes, FiLM conditioning, loss and metric correctness, and a tiny overfit on asymmetric
synthetic CT."""

import pytest
import torch

from ctsynth.seg.losses import confusion_matrix, dice_ce_loss, dice_iou_from_confusion
from ctsynth.seg.unet3d import UNet3D, normalise_hu


def test_shapes_and_divisibility():
    net = UNet3D(1, 5, base=4, depth=2)
    assert net(torch.randn(2, 1, 8, 12, 16)).shape == (2, 5, 8, 12, 16)  # distinct axis lengths


def test_conditioning_contract_and_effect():
    net = UNet3D(1, 3, base=4, depth=2, cond_dim=2)
    x = torch.randn(1, 1, 8, 8, 8)
    with pytest.raises(ValueError):
        net(x)
    with pytest.raises(ValueError):
        UNet3D(1, 3, base=4, depth=2)(x, torch.zeros(1, 2))
    a = net(x, torch.zeros(1, 2))
    for blk in [
        net.bottleneck,
        *net.dec,
    ]:  # FiLM starts as identity; perturb so the metadata must matter
        torch.nn.init.normal_(blk.film.weight, std=0.5)
    assert not torch.allclose(net(x, torch.zeros(1, 2)), net(x, torch.ones(1, 2)))
    assert a.shape == (1, 3, 8, 8, 8)


def test_dice_loss_and_metrics():
    t = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    t[:, :2] = 1
    perfect = torch.nn.functional.one_hot(t, 2).permute(0, 4, 1, 2, 3).float() * 40
    assert float(dice_ce_loss(perfect, t)) < 1e-3
    assert float(dice_ce_loss(-perfect, t)) > 1.0
    cm = confusion_matrix(t, t, 3)
    d, i = dice_iou_from_confusion(cm)
    assert d[:2].tolist() == [1.0, 1.0] and torch.isnan(d[2]) and i[1] == 1.0
    wrong = confusion_matrix(1 - t, t, 2)
    assert dice_iou_from_confusion(wrong)[0].tolist() == [0.0, 0.0]


def test_normalise_hu_clips():
    assert normalise_hu(torch.tensor([-5000.0, 0.0, 9000.0])).tolist() == pytest.approx(
        [-1.024, 0.0, 1.5]
    )


def test_overfits_a_fixed_patch():
    torch.manual_seed(0)
    ct = torch.randn(1, 1, 16, 16, 16) * 0.05
    lab = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    lab[:, 4:12, 3:9, 5:14] = 1
    lab[:, 2:6, 10:15, 2:7] = 2
    ct = ct + lab[:, None].float() * 0.5
    net = UNet3D(1, 3, base=8, depth=2)
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = dice_ce_loss(net(ct), lab)
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < 0.3 * first
