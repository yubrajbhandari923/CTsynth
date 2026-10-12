"""Streaming loader: per-case equality with direct indexing, memmap-only access, canonical order."""

import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from ctsynth.aapm import schema as S
from ctsynth.aapm import synthetic as Y
from ctsynth.aapm.loader import CaseStore
from ctsynth.aapm.schema import ModelTable, SchemaError


def make_dir(tmp_path: Path, tables: dict[int, ModelTable]) -> tuple[Path, Y.SyntheticData]:
    d = Y.generate(Y.SyntheticConfig(n_cases=6, n_pix=16, n_views=16, n_det=24, seed=2), tables)
    Y.write_synthetic_dir(d, tmp_path)
    return tmp_path, d


def test_case_equals_direct_indexing(tmp_path: Path, tables: dict[int, ModelTable]) -> None:
    root, d = make_dir(tmp_path, tables)
    store = CaseStore(root, d.geometry)
    assert len(store) == 6
    for i in (0, 3, 5):
        c = store[i]
        assert c.index == i
        assert c.phantom.shape == (3, 16, 16) and c.phantom.dtype == np.float32
        for j in range(3):  # canonical order: [adipose, fibroglandular, calcification]
            assert np.array_equal(c.phantom[j], d.phantoms[i, j])
        for k in (50, 80):
            assert np.array_equal(c.transmission[k], d.transmission[k][i])
            assert np.array_equal(c.image[k], d.images[k][i])


def test_iter_cases_order_and_bounds(tmp_path: Path, tables: dict[int, ModelTable]) -> None:
    root, d = make_dir(tmp_path, tables)
    store = CaseStore(root, d.geometry)
    assert [c.index for c in store.iter_cases()] == list(range(6))
    assert [c.index for c in store.iter_cases([4, 1])] == [4, 1]
    for bad in (-1, 6):
        with pytest.raises(IndexError):
            store[bad]


def test_uses_readonly_memmaps_and_copies_out(
    tmp_path: Path, tables: dict[int, ModelTable], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, d = make_dir(tmp_path, tables)
    calls: list[dict[str, object]] = []
    real_load = np.load

    def spy(*a: object, **kw: object) -> object:
        calls.append(kw)
        return real_load(*a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(np, "load", spy)
    store = CaseStore(root, d.geometry)
    list(store.iter_cases())
    assert calls and all(kw.get("mmap_mode") == "r" for kw in calls)  # never a full read
    assert len(store.arrays) == 7
    for name, arr in store.arrays.items():
        assert isinstance(arr, np.memmap) and not arr.flags.writeable, name
    c = store[0]
    for arr in (c.phantom, *c.transmission.values(), *c.image.values()):
        assert not isinstance(arr, np.memmap)
        assert not any(np.shares_memory(arr, m) for m in store.arrays.values())  # copies


def test_peak_memory_is_order_one_case(tmp_path: Path, tables: dict[int, ModelTable]) -> None:
    n_case = 24
    d = Y.generate(Y.SyntheticConfig(n_case, 64, 8, 16, seed=0), tables)
    Y.write_synthetic_dir(d, tmp_path)
    total = sum(a.nbytes for a in (d.phantoms, *d.images.values(), *d.transmission.values()))
    store = CaseStore(tmp_path, d.geometry)
    tracemalloc.start()
    try:
        for _ in store.iter_cases():  # case discarded each step
            pass
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # one case is 1/24 of the data; 0.35 leaves ~8x headroom for temporaries yet fails any
    # implementation that holds even a third of the dataset (mmap pages are not Python allocations)
    assert peak < 0.35 * total


def test_swapped_material_files_detected(tmp_path: Path, tables: dict[int, ModelTable]) -> None:
    """On-disk Adipose <-> Calcification swap must not be silently accepted."""
    root, d = make_dir(tmp_path, tables)
    a, c = root / "Phantom_Adipose.npy", root / "Phantom_Calcification.npy"
    tmp = root / "swap.npy"
    a.rename(tmp)
    c.rename(a)
    tmp.rename(c)
    with pytest.raises(SchemaError) as e:
        CaseStore(root, d.geometry)[0]
    assert e.value.convention == S.CONV_MATERIAL_ORDER


def test_open_errors(tmp_path: Path, tables: dict[int, ModelTable]) -> None:
    root, d = make_dir(tmp_path, tables)
    wrong = S.GeometrySpec(32, 18.0, 16, 24)
    with pytest.raises(SchemaError) as e:
        CaseStore(root, wrong)
    assert e.value.convention == S.CONV_SHAPE
    gz = tmp_path / "gz"
    gz.mkdir()
    (gz / "Phantom_Adipose.npy.gz").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match=r"Phantom_Adipose\.npy"):
        CaseStore(gz, d.geometry)


REAL = Path(__file__).resolve().parents[2] / "data" / "aapm"


@pytest.mark.skipif(not (REAL / "Phantom_Adipose.npy").exists(), reason="real AAPM arrays absent")
def test_real_data_first_case_valid() -> None:  # optional; skipped in CI
    c = CaseStore(REAL)[0]
    S.validate_phantoms(c.phantom[None], S.REAL_GEOMETRY, material_names=S.MATERIALS)
