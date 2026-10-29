"""Small measurement helpers: CUDA-synchronised timing with repetitions, sampled device-memory
peaks, host RSS, file-cache control."""

from __future__ import annotations

import os
import resource
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch

MIB = 2.0**20


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def device_used_bytes() -> int:
    free, total = torch.cuda.mem_get_info()
    return int(total - free)


def host_rss_peak_mib() -> float:
    """Peak resident set size of this process in MiB (Linux ru_maxrss is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class MemSampler:
    """Polls cudaMemGetInfo every `dt` seconds: peak device memory in use (includes the CUDA
    context and non-PyTorch allocations
    such as LEAP's). A lower bound of the true peak."""

    def __init__(self, dt: float = 0.002) -> None:
        self.dt, self.base, self.peak = dt, 0, 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self) -> MemSampler:
        sync()
        self.base = self.peak = device_used_bytes()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, device_used_bytes())
            time.sleep(self.dt)

    def __exit__(
        self, *exc: tuple[type[BaseException], BaseException, TracebackType] | Any
    ) -> None:
        sync()
        self.peak = max(self.peak, device_used_bytes())
        self._stop.set()
        if self._t is not None:
            self._t.join()


@dataclass
class Timing:
    op: str
    times_s: list[float]
    peak_device_mib: float | None = None

    def summary(self) -> dict[str, Any]:
        t = self.times_s
        return {
            "op": self.op,
            "n": len(t),
            "median_s": statistics.median(t),
            "mean_s": statistics.fmean(t),
            "std_s": statistics.pstdev(t) if len(t) > 1 else 0.0,
            "min_s": min(t),
            "max_s": max(t),
            "peak_device_mib": self.peak_device_mib,
        }


def timed(
    op: str, fn: Callable[[], Any], reps: int = 7, warmup: int = 1, memory: bool = False
) -> tuple[Timing, Any]:
    """Run `fn` `warmup` + `reps` times with CUDA syncs around each timed call; optionally one
    extra sampled run for the memory peak.
    Returns the Timing and the result of the last timed call."""
    out = None
    for _ in range(warmup):
        out = fn()
        sync()
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        ts.append(time.perf_counter() - t0)
    peak = None
    if memory and torch.cuda.is_available():
        with MemSampler() as ms:
            out = fn()
        peak = ms.peak / MIB
    return Timing(op, ts, peak), out


def drop_file_cache(path: str) -> None:
    """Ask the kernel to evict a file's pages (posix_fadvise DONTNEED; works without root for files
    we can open)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
