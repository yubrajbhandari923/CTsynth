"""Aggregate benchmark_leap.py raw outputs into CSV tables, plots and a metadata JSON (reads OUT/benchmarks/raw, writes OUT/benchmarks).

Usage: benchmark_aggregate.py [BENCH_DIR]   (default /home/yb107/datasets/truect_pilot/outputs/benchmarks)
Writes: benchmark_results.csv (every timed sample), benchmark_stats.csv, benchmark_estimates.csv, benchmark_config.json, plots/*.png|pdf.
benchmark_summary.md is written by hand from these tables (interpretation is not generated).
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/yb107/datasets/truect_pilot/outputs/benchmarks")
raw = B / "raw"
CONFIGS = ["small", "ref", "copd0"]
LABEL = {"small": "small\n192²×48 @2.4 mm\n3000 views", "ref": "reference\n768²×192 @0.6 mm\n3000 views", "copd0": "dcm_000\n768²×526 @0.6 mm\n9000 views"}
OPS = ["A_forward_project", "B_adjoint_backproject", "C1_FBP_helical_gpu_tensors", "C2_FBP_helical_host_numpy",
       "D1_torch_forward", "D2_torch_backward", "D3_torch_forward_loss_backward", "D4_torch_iteration_plus_Adam"]
SHORT = {"A_forward_project": "A fwd project", "B_adjoint_backproject": "B adjoint BP", "C1_FBP_helical_gpu_tensors": "C1 FBP (GPU tensors)",
         "C2_FBP_helical_host_numpy": "C2 FBP (host arrays)", "D1_torch_forward": "D1 torch fwd", "D2_torch_backward": "D2 torch bwd",
         "D3_torch_forward_loss_backward": "D3 fwd+loss+bwd", "D4_torch_iteration_plus_Adam": "D4 D3+Adam"}

meta = {c: json.load(open(raw / f"meta_checks_{c}.json")) for c in CONFIGS if (raw / f"meta_checks_{c}.json").exists()}
have = list(meta)
for c in have:  # validation rerun with more free VRAM supersedes the in-run checks for the same keys (original kept under *_inrun)
    f_ = raw / f"checks_only_{c}.json"
    if f_.exists():
        new = json.load(open(f_))
        for k, v in new.items():
            if k in meta[c]["checks"]:
                meta[c]["checks"][k + "_inrun"] = meta[c]["checks"][k]
            meta[c]["checks"][k] = v
        meta[c]["checks"]["checks_rerun_file"] = f_.name
rows = pd.concat([pd.read_csv(raw / f"rows_{c}.csv") for c in have])
mem = pd.concat([pd.read_csv(raw / f"mem_{c}.csv") for c in have])
setup = pd.concat([pd.read_csv(raw / f"setup_{c}.csv") for c in have])
for c in have:
    m = meta[c]["meta"]
    rows.loc[rows.config == c, ["n_views", "n_det_samples", "n_voxels"]] = [m["n_views"], m["n_det_samples"], m["n_voxels"]]
rows.to_csv(B / "benchmark_results.csv", index=False)

warm = rows[rows.rep != "cold"].copy()
warm["time_s"] = warm.time_s.astype(float)
st = warm.groupby(["config", "op"]).time_s.agg(n="count", median="median", mean="mean", std="std", min="min", max="max").reset_index()
cold = rows[rows.rep == "cold"][["config", "op", "time_s"]].rename(columns={"time_s": "cold_first_call_s"})
st = st.merge(cold, on=["config", "op"]).merge(mem, on=["config", "op"])
st["n_det_samples"] = st.config.map(lambda c: meta[c]["meta"]["n_det_samples"])
st["n_voxels"] = st.config.map(lambda c: meta[c]["meta"]["n_voxels"])
st["det_samples_per_s_M"] = st.n_det_samples / st["median"] / 1e6
st["voxels_per_s_M"] = st.n_voxels / st["median"] / 1e6
st["views_per_s"] = st.config.map(lambda c: meta[c]["meta"]["n_views"]) / st["median"]
st["op"] = pd.Categorical(st.op, OPS, ordered=True)
st["config"] = pd.Categorical(st.config, CONFIGS, ordered=True)
st = st.sort_values(["config", "op"]).reset_index(drop=True)
st.to_csv(B / "benchmark_stats.csv", index=False)

# ---- iteration-time estimates (clearly labelled) ----
est = []
for c in have:
    s = st[st.config == c].set_index("op")
    su = setup[setup.config == c].set_index("stage").time_s
    if "D4_torch_iteration_plus_Adam" not in s.index or "D3_torch_forward_loss_backward" not in s.index:
        continue  # an autograd op ran out of memory for this config: no estimate (reported in the summary)
    d3, d4 = float(s.loc["D3_torch_forward_loss_backward", "median"]), float(s.loc["D4_torch_iteration_plus_Adam", "median"])
    d_real = max(d3, d4)  # D4 can come out faster than D3 (copd0: 40.1 vs 43.3 s, unexplained): take the slower, do not credit the noise
    fa, ba = float(s.loc["A_forward_project", "median"]), float(s.loc["B_adjoint_backproject", "median"])
    reload_s = float(su["disk_read_and_convert_projections"] + su["H2D_projections"])
    for n in (100, 1000):
        est.append(dict(config=c, iterations=n, kind="idealized: N x (A + A^T kernel time only)", seconds=n * (fa + ba)))
        est.append(dict(config=c, iterations=n, kind="measured-iteration: N x D3 (fwd+loss+bwd, volume resident)", seconds=n * d3))
        est.append(dict(config=c, iterations=n, kind="realistic A: N x max(D3,D4) (adds Adam step; same volume/projections reused)", seconds=n * d_real))
        est.append(dict(config=c, iterations=n, kind="realistic B: N x (max(D3,D4) + re-read & upload this scan's projections each iteration)", seconds=n * (d_real + reload_s)))
est = pd.DataFrame(est)
est["minutes"] = est.seconds / 60
est.to_csv(B / "benchmark_estimates.csv", index=False)

env = {c: json.load(open(raw / f"env_{c}.json")) for c in have if (raw / f"env_{c}.json").exists()}
json.dump(dict(environment=env, per_config=meta, setup_timings=setup.to_dict("records"),
               memory_method="cudaMemGetInfo sampled every 2 ms in a thread during separate profiled runs (lower bound of true peak) + torch.cuda.max_memory_allocated",
               timing_method="time.perf_counter around op with torch.cuda.synchronize() before/after; 2 warm-up calls; cold first call reported separately",
               ), open(B / "benchmark_config.json", "w"), indent=1, default=str)

# ---- plots ----
(B / "plots").mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
cols = {"small": "#8fb8de", "ref": "#2f6fa8", "copd0": "#0c2f57"}


def grouped(metric, err, ylabel, fname, log, hline=None):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.8 / len(have)
    for i, c in enumerate(have):
        s = st[st.config == c].set_index("op").reindex(OPS)
        x = np.arange(len(OPS)) + (i - (len(have) - 1) / 2) * w
        e = None if err is None else np.vstack([s[metric] - s[err[0]], s[err[1]] - s[metric]])
        ax.bar(x, s[metric], w, yerr=e, capsize=2, color=cols[c], label=LABEL[c].replace("\n", ", "), lw=0)
    ax.set_xticks(range(len(OPS)))
    ax.set_xticklabels([SHORT[o] for o in OPS], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    if hline:
        ax.axhline(hline[0], color="#b03a2e", ls="--", lw=1, label=hline[1])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(B / "plots" / f"{fname}.{ext}")
    plt.close(fig)


grouped("median", ("min", "max"), "execution time per call [s] (median, min–max bars)", "time_by_operation", True)
grouped("peak_device_MiB", None, "peak GPU memory in use [MiB]", "peak_gpu_memory_by_operation", False, hline=(24564, "RTX A5000 capacity (24564 MiB)"))
fig, ax = plt.subplots(figsize=(6, 3.8))
for c in have:
    s = est[(est.config == c) & est.kind.str.startswith("realistic A")]
    ax.plot(s.iterations, s.minutes, "o-", color=cols[c], label=LABEL[c].split("\n")[0] + " (max(D3,D4) × N)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("iterations"); ax.set_ylabel("estimated wall time [min]")
ax.legend(frameon=False, fontsize=8); fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(B / "plots" / f"iteration_time_estimate.{ext}")
print(st.drop(columns=["n_det_samples", "n_voxels"]).round(4).to_string())
print(est.round(2).to_string())
print(setup.round(3).to_string())
