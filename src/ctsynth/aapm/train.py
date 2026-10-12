"""Deterministic image-domain training loops (spec: docs/aapm/unsupervised-spec.md).

`train_label_free` takes ONLY measured images: it has no parameter through which ground truth
could enter (structural guard). `train_supervised` is a separate function taking labels explicitly.
Early stopping uses validation image-domain consistency (a measurement-domain quantity), never
truth, and the best-validation weights are restored. Determinism: model init under a forked RNG
seeded by `seed`; batch order from an explicit torch.Generator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from ctsynth.aapm.losses import compose, consistency_image
from ctsynth.aapm.torch_forward import AapmTorchForward

History = dict[str, list[float]]


def _fit(
    factory: Callable[[], nn.Module],
    x_tr: torch.Tensor,
    x_va: torch.Tensor,
    loss_fn: Callable[..., tuple[torch.Tensor, dict[str, float]]],
    val_fn: Callable[[nn.Module], float],
    *,
    idx_extra: tuple[torch.Tensor, ...],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    patience: int | None,
) -> tuple[nn.Module, History]:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = factory()
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist: History = {"train_total": [], "val_consistency": []}
    best, best_state, bad = float("inf"), None, 0
    n = x_tr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot, cnt = 0.0, 0
        parts_sum: dict[str, float] = {}
        for s in range(0, n, batch_size):
            ix = perm[s : s + batch_size]
            opt.zero_grad()
            loss, parts = loss_fn(model, x_tr[ix], *(e[ix] for e in idx_extra))
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            tot += float(loss.detach()) * len(ix)
            cnt += len(ix)
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v * len(ix)
        hist["train_total"].append(tot / cnt)
        for k, v in parts_sum.items():
            hist.setdefault(f"train_{k}", []).append(v / cnt)
        model.eval()
        with torch.no_grad():
            v = val_fn(model)
        hist["val_consistency"].append(v)
        if v < best:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if patience is not None and bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def train_label_free(
    factory: Callable[[], nn.Module],
    fwd: AapmTorchForward,
    meas_train: torch.Tensor,
    meas_val: torch.Tensor,
    weights: Mapping[str, float],
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 4,
    seed: int = 0,
    patience: int | None = None,
    air_cutoff: float | None = 0.1,
    term_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[nn.Module, History]:
    """Train on measured images (N,2,H,W) only. `weights` picks registry terms among
    consistency_image, nonneg, sum_to_one, tv, entropy (each independently weighted).

    sum_to_one is applied to TISSUE pixels only, where tissue = NOT(both measured
    images < air_cutoff), a measurement-derived mask (None = everywhere). Validation =
    consistency_image on meas_val.
    """

    def mask_of(m: torch.Tensor) -> torch.Tensor | None:
        return None if air_cutoff is None else ~((m < air_cutoff).all(dim=1))

    def loss_fn(model: nn.Module, m: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        f = model(m)
        args = {
            "consistency_image": (f, m, fwd),
            "nonneg": (f,),
            "sum_to_one": (f, mask_of(m)),
            "tv": (f,),
            "entropy": (f,),
        }
        return compose(weights, args, term_kwargs)

    def val_fn(model: nn.Module) -> float:
        return float(consistency_image(model(meas_val), meas_val, fwd))

    return _fit(
        factory,
        meas_train,
        meas_val,
        loss_fn,
        val_fn,
        idx_extra=(),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        patience=patience,
    )


def train_supervised(
    factory: Callable[[], nn.Module],
    meas_train: torch.Tensor,
    truth_train: torch.Tensor,
    meas_val: torch.Tensor,
    truth_val: torch.Tensor,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 4,
    seed: int = 0,
    patience: int | None = None,
) -> tuple[nn.Module, History]:
    """Labelled MSE baseline: mean (model(meas) - truth)^2. Validation is the same MSE on val labels
    (history key stays 'val_consistency' for plotting symmetry; here it is label MSE)."""

    def loss_fn(
        model: nn.Module, m: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        v = ((model(m) - t) ** 2).mean()
        return v, {"mse": float(v.detach())}

    def val_fn(model: nn.Module) -> float:
        return float(((model(meas_val) - truth_val) ** 2).mean())

    return _fit(
        factory,
        meas_train,
        meas_val,
        loss_fn,
        val_fn,
        idx_extra=(truth_train,),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        patience=patience,
    )
