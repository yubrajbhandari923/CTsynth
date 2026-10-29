"""Timing helper: repetition counts, statistics, cache-drop on a real file."""

import numpy as np

from ctsynth.xcat.profiling import drop_file_cache, host_rss_peak_mib, timed


def test_timed_counts_and_summary():
    calls = []
    t, out = timed("op", lambda: calls.append(1) or len(calls), reps=5, warmup=2)
    assert len(calls) == 7 and len(t.times_s) == 5 and out == 7
    s = t.summary()
    assert (
        s["n"] == 5 and s["min_s"] <= s["median_s"] <= s["max_s"] and s["peak_device_mib"] is None
    )


def test_rss_and_cache_drop(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(np.zeros(1000, np.uint8).tobytes())
    drop_file_cache(str(p))
    assert host_rss_peak_mib() > 10
