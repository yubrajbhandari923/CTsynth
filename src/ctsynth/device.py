"""Device choice: CPU by default; MPS/CUDA only when explicitly requested and available."""


def pick_device(requested: str = "cpu") -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(f"device={requested!r} needs the 'torch' extra") from e
    ok = {"mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    if requested not in ok:
        raise ValueError(f"unknown device {requested!r}")
    if not ok[requested]:
        raise RuntimeError(f"device {requested!r} requested but unavailable")
    return requested
