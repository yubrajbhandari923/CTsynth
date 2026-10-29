"""Variant grouping, conditioning vector, deterministic combinations, channel apodisation."""

import pytest
import torch

from ctsynth.xcat.variants import (
    VariantSpec,
    apodise_channels,
    family_variants,
    sample_combined,
    sample_training_variant,
    spec_from_dict,
    spec_to_dict,
)


def test_groups_share_projections_only_when_geometry_and_energy_match():
    v = family_variants()
    assert (
        v["clean"].group_key() == v["n0_1e5"].group_key() == v["apod_1p5"].group_key()
    )  # noise / kernel reuse one forward projection
    keys = {v[k].group_key() for k in ("e60", "e80", "pitch1p4", "dec2", "clean")}
    assert len(keys) == 5  # every acquisition change needs its own projections
    assert {x.family for x in v.values()} == {"A", "B", "C", "D"}


def test_cond_vector_and_combined_sampling_deterministic_and_bounded():
    assert (
        len(VariantSpec("x", "A").cond_vector()) == 7
        and VariantSpec("x", "A").cond_vector()[0] == 0.0
    )
    a, b, c = sample_combined(7, 3), sample_combined(7, 3), sample_combined(7, 4)
    assert a == b and a != c
    for i in range(50):
        s = sample_combined(0, i)
        assert (
            3e4 <= s.n0 <= 1e6
            and s.acq.energy_kev in (60.0, 70.0, 80.0)
            and s.acq.pitch in (1.0, 1.2, 1.4)
            and s.apod_sigma_px in (0.0, 0.75)
            and s.lowpass_w in (1.0, 2.0, 4.0)
        )


def test_apodisation_properties():
    g = torch.Generator().manual_seed(0)
    white = torch.randn(40, 4, 200, generator=g)
    flat = torch.full((10, 2, 50), 3.0)
    assert torch.equal(apodise_channels(white.clone(), 0.0), white)
    assert torch.allclose(
        apodise_channels(flat.clone(), 1.5), flat, atol=1e-5
    )  # constants preserved (unit-sum kernel, replicate padding)
    sm = apodise_channels(white.clone(), 1.5)
    assert float(sm.var()) < 0.25 * float(white.var())
    chunked = apodise_channels(white.clone(), 1.5, view_chunk=7)
    assert torch.allclose(chunked, sm, atol=1e-6)
    with pytest.raises(ValueError):
        apodise_channels(torch.zeros(3, 3), 1.0)


def test_training_variant_never_uses_held_out_settings_and_round_trips():
    seen = set()
    for k in range(300):
        s = sample_training_variant(0, k)
        assert s.acq.energy_kev in (60.0, 70.0) and s.acq.pitch == 1.0
        assert s.acq.view_decimation in (1, 2)
        assert 3e4 <= s.n0 <= 1e6 and s.lowpass_w in (1.0, 2.0, 4.0)
        assert spec_from_dict(spec_to_dict(s)) == s
        seen.add((s.acq.energy_kev, s.acq.view_decimation, s.lowpass_w))
    a, b, c = (
        sample_training_variant(0, 5),
        sample_training_variant(0, 5),
        sample_training_variant(0, 6),
    )
    assert a == b and a != c
    assert len(seen) >= 10  # real diversity across patients
