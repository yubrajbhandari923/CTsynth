"""Class-balanced patch sampling: shapes, determinism, foreground bias, alignment of CT and labels
(asymmetric volume)."""

import numpy as np
import torch

from ctsynth.seg.dataset import PatchDataset, VolumeRecord, class_voxel_samples
from ctsynth.xcat.cache import write_hu
from ctsynth.xcat.cases import find_copd_case, list_copd_cases


def _record(tmp_path):
    lab = np.zeros((40, 48, 56), np.uint8)
    lab[10:20, 12:30, 30:50] = 1
    lab[25:35, 5:15, 5:15] = 2
    hu = (lab.astype(np.float32) * 300.0) - 1000.0  # HU encodes the label: alignment is checkable
    w = write_hu(tmp_path / "ct", hu, "npy_i16q")
    np.save(tmp_path / "lab.npy", lab)
    return VolumeRecord(
        "p0",
        w.path,
        "npy_i16q",
        str(tmp_path / "lab.npy"),
        (1.0, 2.0),
        class_voxel_samples(lab, 500),
    ), lab


def test_patch_alignment_determinism_and_foreground_bias(tmp_path):
    rec, lab = _record(tmp_path)
    ds = PatchDataset([rec], 16, 64, seed=3, p_foreground=1.0)
    x, y, c = ds[5]
    assert (
        x.shape == (1, 16, 16, 16)
        and y.shape == (16, 16, 16)
        and y.dtype == torch.int64
        and c.tolist() == [1.0, 2.0]
    )
    assert torch.allclose(
        x[0] * 1000.0, y.float() * 300.0 - 1000.0, atol=0.2
    )  # CT and labels cover the same voxels
    assert torch.equal(ds[5][0], PatchDataset([rec], 16, 64, seed=3)[5][0])
    assert (
        np.mean([float((ds[i][1] > 0).any()) for i in range(40)]) == 1.0
    )  # p_foreground=1 -> always contains an object
    ds.epoch = 1
    assert not torch.equal(ds[5][0], x)


def test_class_samples_cover_all_present_classes(tmp_path):
    rec, lab = _record(tmp_path)
    assert sorted(rec.class_samples) == [0, 1, 2]
    k = rec.class_samples[1]
    assert (lab[k[:, 0], k[:, 1], k[:, 2]] == 1).all()


def test_case_discovery(tmp_path):
    (tmp_path / "material_tables").mkdir()
    (tmp_path / "material_lac" / "123").mkdir(parents=True)
    n = tmp_path / "phantom_123_30_20_10_8bit_textured_copd_eroded_flipped.bin"
    n.write_bytes(b"\0" * 6000)
    assert list_copd_cases(tmp_path) == []  # tables missing -> incomplete
    (tmp_path / "material_tables" / "challenge_materials_copd_smeared_123.xlsx").write_bytes(b"x")
    for f in ("lin_att_120kV_121materials.bin", "conversion_table_120kV_121materials.in"):
        (tmp_path / "material_lac" / "123" / f).write_bytes(b"x")
    c = find_copd_case(tmp_path, "123")
    assert c.shape_zyx == (10, 20, 30) and len(list_copd_cases(tmp_path)) == 1
