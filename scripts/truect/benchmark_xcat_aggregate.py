"""Aggregate benchmark_xcat.py raw output + e2e pipeline metadata into CSVs, plots and a config JSON.  usage: benchmark_xcat_aggregate.py BENCH_DIR
benchmark_summary.md is written by hand from these tables (interpretation is not generated)."""
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B = Path(sys.argv[1]); raw = B / "raw"
OPS = ["S1_load_materials_and_native_mu_conversion", "S2_resample_to_grid", "A_forward_project", "B_adjoint_backproject", "C_helical_FBP",
       "D1_autograd_forward", "D2_autograd_backward", "D3_autograd_forward_loss_backward", "D4_autograd_iteration_plus_Adam"]
SHORT = dict(zip(OPS, ["S1 load+mu", "S2 resample", "A forward", "B adjoint", "C FBP", "D1 ag fwd", "D2 ag bwd", "D3 fwd+loss+bwd", "D4 +Adam"]))
metas = {Path(f).name.replace("_meta.json", ""): json.load(open(f)) for f in sorted(glob.glob(str(raw / "*_meta.json")))}
tags = sorted(metas, key=lambda t: (["reference", "copd_22728", "copd_21535"].index(metas[t]["meta"]["case"]), -metas[t]["meta"]["vox_mm"]))
rows = pd.concat([pd.read_csv(raw / f"{t}_rows.csv") for t in tags if (raw / f"{t}_rows.csv").exists()])
mem = pd.concat([pd.read_csv(raw / f"{t}_mem.csv") for t in tags if (raw / f"{t}_mem.csv").exists()])
rows.to_csv(B / "benchmark_results.csv", index=False)
warm = rows[rows.rep != "cold"].copy(); warm["time_s"] = warm.time_s.astype(float)
st = warm.groupby(["config", "op"]).time_s.agg(n="count", median="median", mean="mean", std="std", min="min", max="max").reset_index()
st = st.merge(rows[rows.rep == "cold"][["config", "op", "time_s"]].rename(columns={"time_s": "cold_first_call_s"}), on=["config", "op"]).merge(mem, on=["config", "op"])
for k in ("case", "vox_mm", "n_views", "n_voxels", "n_det_samples", "native_voxels", "proj_GiB", "vol_GiB"):
    st[k] = st.config.map(lambda c: metas[c]["meta"][k])
st["vram_MiB"] = st.config.map(lambda c: metas[c]["meta"]["vram_MiB"])
def dims(r):
    m = metas[r.config]["meta"]; g, p, n = m["grid_zyx"], m["proj_shape"], m["native_shape"]
    return {"S1": (f"native labels {n}", "native mu (reduced)"), "S2": (f"native labels {n}", f"grid {g}"), "A": (f"vol {g}", f"proj {p}"),
            "B": (f"proj {p}", f"vol {g}"), "C": (f"proj {p}", f"vol {g}"), "D": (f"vol {g}", f"proj {p}, grad vol {g}")}[r.op[0 if r.op[0] != "D" else 0] if r.op[0] in "ABCD" else r.op[:2]]
st["input"], st["output"] = zip(*st.apply(dims, axis=1))
st["throughput"] = st.apply(lambda r: (f"{r.native_voxels / r['median'] / 1e6:.1f} M native vox/s" if r.op[0] == "S" else
                                       f"{r.n_det_samples / r['median'] / 1e6:.1f} M det samples/s; {r.n_views / r['median']:.0f} views/s" if r.op[0] in "AD" else
                                       f"{r.n_voxels / r['median'] / 1e6:.1f} M vox/s; {r.n_views / r['median']:.0f} views/s"), axis=1)
st["op"] = pd.Categorical(st.op, OPS, ordered=True)
st["config"] = pd.Categorical(st.config, tags, ordered=True)
st = st.sort_values(["config", "op"]).reset_index(drop=True)
st.to_csv(B / "benchmark_stats.csv", index=False)

# prior (reconstructed-mu volume, 9000-view dcm_000, 310 M voxels, A5000) for comparison
prior = pd.DataFrame([dict(op="A_forward_project", prior_s=6.865), dict(op="B_adjoint_backproject", prior_s=36.24), dict(op="C_helical_FBP", prior_s=15.71),
                      dict(op="D3_autograd_forward_loss_backward", prior_s=43.29)]); prior["prior_views"] = 9000; prior["prior_voxels"] = 768 * 768 * 526
cmp_rows = []
for _, r in st.iterrows():
    p = prior[prior.op == r.op]
    if len(p):
        p = p.iloc[0]
        cmp_rows.append(dict(config=r.config, op=r.op, median_s=r["median"], prior_s=p.prior_s, ratio_to_prior=r["median"] / p.prior_s, views=r.n_views, voxels=r.n_voxels,
                             prior_views=p.prior_views, prior_voxels=p.prior_voxels,
                             norm_time_per_view_voxel_ns=r["median"] / (r.n_views * r.n_voxels) * 1e9, prior_norm_ns=p.prior_s / (p.prior_views * p.prior_voxels) * 1e9))
pd.DataFrame(cmp_rows).to_csv(B / "benchmark_comparison_to_prior.csv", index=False)

# end-to-end pipeline runs
e2e = []
for f in sorted(glob.glob(str(B / "e2e/*/metadata.json"))):
    m = json.load(open(f)); t = m["timings_s"]
    stages = {k: v for k, v in t.items() if k != "total_wall_clock_s"}
    e2e.append(dict(case=m["case"], vox_mm=m["args"]["voxel_size_mm"], total_s=t["total_wall_clock_s"], **stages, other_s=t["total_wall_clock_s"] - sum(stages.values()),
                    forward_s=m["forward_projection_s"], fbp_s=m["fbp_s"], peak_fwd_MiB=m["peak_gpu_MiB"]["forward"], peak_fbp_MiB=m["peak_gpu_MiB"]["fbp"],
                    body_hu_mae=m["metrics"]["body_eroded2"]["hu_mae"], body_hu_rmse=m["metrics"]["body_eroded2"]["hu_rmse"], lung_hu_mae=(m["metrics"]["lung_eroded2"] or {}).get("hu_mae"),
                    xcorr_shift_vox=m["registration"]["xcorr_shift_voxels_zyx_recon_vs_ideal"]))
e2e = pd.DataFrame(e2e)
if len(e2e):
    e2e.to_csv(B / "benchmark_e2e.csv", index=False)

pk = st.pivot_table(index="config", columns="op", values="peak_device_MiB", observed=True)
# memory feasibility: EXACT count of tensors that must be resident (lower bound, no fit, no extrapolation of LEAP internals).
# A forward: mu volume + projections. B adjoint: projections + output volume. C FBP: + the input copy FBP makes (inplace=False).
# D autograd step: parameter x, grad(x), leaptorch vol buffer (3 volumes) + projection buffer, target, (pred-y) and its square/grad (>=4 projection tensors).
BOUND = {"A_forward_project": (1, 1), "B_adjoint_backproject": (1, 1), "C_helical_FBP": (1, 2), "D3_autograd_forward_loss_backward": (3, 4)}
feas = []
cfgs = [(c, metas[c]["meta"]["case"], metas[c]["meta"]["vox_mm"], metas[c]["meta"]["vol_GiB"] * 1024, metas[c]["meta"]["proj_GiB"] * 1024, metas[c]["meta"]["vram_MiB"]) for c in tags]
for case in ["reference", "copd_22728", "copd_21535"]:
    m0 = [metas[c]["meta"] for c in tags if metas[c]["meta"]["case"] == case][0]
    cfgs.append((f"{case}_native0.25mm", case, 0.25, m0["native_voxels"] * 4 / 2**20, m0["proj_GiB"] * 1024, m0["vram_MiB"]))
for cfg, case, vox, vol, proj, vram in cfgs:
    for op, (nv, npj) in BOUND.items():
        bound = nv * vol + npj * proj
        meas = pk.loc[cfg, op] if (cfg in pk.index and op in pk.columns) else np.nan
        feas.append(dict(config=cfg, case=case, vox_mm=vox, op=op, vol_MiB=vol, proj_MiB=proj, resident_lower_bound_MiB=bound, measured_peak_MiB=meas, vram_MiB=vram,
                         verdict=("measured" if not np.isnan(meas) else ("lower bound exceeds VRAM: cannot run" if bound > vram else "lower bound fits; true peak adds LEAP/torch buffers (unmeasured at this size): NOT attempted")),
                         note="resident-tensor lower bound = exact count, not a model of LEAP internals"))
pd.DataFrame(feas).to_csv(B / "benchmark_native_feasibility.csv", index=False)
fit_rows = []

# training-loop estimates (ESTIMATES from measured per-iteration times)
est = []
for c in tags:
    s = st[st.config == c].set_index("op")
    if "D3_autograd_forward_loss_backward" not in s.index or "D4_autograd_iteration_plus_Adam" not in s.index:
        continue
    it = max(float(s.loc["D3_autograd_forward_loss_backward", "median"]), float(s.loc["D4_autograd_iteration_plus_Adam", "median"]))
    m = metas[c]["meta"]; peak = float(s.loc["D4_autograd_iteration_plus_Adam", "peak_device_MiB"])
    free = m["vram_MiB"] - peak
    cnn = 16 * 8 * 4 * 2 * m["n_voxels"] / 2**20  # 16 channels x 8 layers x fp32 x (activation + grad), full-volume 3D CNN (illustrative assumption)
    est.append(dict(config=c, s_per_iteration=it, iterations_per_hour=3600 / it, t_100_min=100 * it / 60, t_1000_h=1000 * it / 3600, peak_D4_MiB=peak, free_vram_for_network_MiB=free,
                    illustrative_full_volume_3D_CNN_16ch_8layers_MiB=cnn, fits_with_network=bool(free > cnn)))
pd.DataFrame(est).to_csv(B / "benchmark_training_estimates.csv", index=False)

json.dump(dict(meta={t: metas[t]["meta"] for t in tags}, checks={t: metas[t]["checks"] for t in tags}, grad_checks={Path(f).stem: json.load(open(f)) for f in sorted(glob.glob(str(raw / "*_gradchecks.json")))},
               timing_method="time.perf_counter around op with torch.cuda.synchronize() before/after; warm-ups then reps; cold first call reported separately",
               memory_method="cudaMemGetInfo sampled every 2 ms in separate profiled runs (lower bound, includes CUDA context) + torch allocator peak",
               e2e=e2e.to_dict("records") if len(e2e) else []), open(B / "benchmark_config.json", "w"), indent=1, default=str)

(B / "plots").mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
cm = plt.get_cmap("tab20")
def grouped(metric, ylabel, name, log, hline=None, err=True):
    fig, ax = plt.subplots(figsize=(11, 4.4)); w = 0.8 / len(tags)
    for i, c in enumerate(tags):
        s = st[st.config == c].set_index("op").reindex(OPS)
        x = np.arange(len(OPS)) + (i - (len(tags) - 1) / 2) * w
        e = np.vstack([s[metric] - s["min"], s["max"] - s[metric]]) if err and metric == "median" else None
        ax.bar(x, s[metric], w, yerr=e, capsize=1.5, color=cm(i), label=c.replace("_", " "), lw=0)
    ax.set_xticks(range(len(OPS))); ax.set_xticklabels([SHORT[o] for o in OPS], rotation=25, ha="right"); ax.set_ylabel(ylabel)
    if log: ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    for i, c in enumerate(tags):  # ops that ran out of GPU memory have no measurement: say so on the plot
        s = st[st.config == c].set_index("op").reindex(OPS)
        for j, o in enumerate(OPS):
            if np.isnan(s.loc[o, metric]):
                ax.text(j + (i - (len(tags) - 1) / 2) * w, lo * (1.3 if log else 1) + (0 if log else 0.01 * hi), "OOM", rotation=90, ha="center", va="bottom", color="#b03a2e", fontsize=7)
    if hline: ax.axhline(hline[0], color="#b03a2e", ls="--", lw=1, label=hline[1])
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left"); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(B / "plots" / f"{name}.{ext}")
    plt.close(fig)
grouped("median", "execution time per call [s] (median, min-max)", "time_by_stage", True)
grouped("peak_device_MiB", "peak GPU memory in use [MiB]", "peak_gpu_memory", False, hline=(float(st.vram_MiB.iloc[0]), "GPU capacity"))
if len(e2e):
    fig, ax = plt.subplots(figsize=(9, 4)); cols = [c for c in e2e.columns if c.endswith("_s") and c[0].isdigit()] + ["other_s"]
    lab = [f"{r.case}\n{r.vox_mm} mm" for r in e2e.itertuples()]; bot = np.zeros(len(e2e))
    for j, c in enumerate(cols):
        ax.bar(lab, e2e[c], bottom=bot, label=c, color=cm(j), lw=0); bot += e2e[c].values
    ax.set_ylabel("seconds"); ax.legend(frameon=False, fontsize=7); ax.set_title("end-to-end pipeline wall clock by stage"); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(B / "plots" / f"e2e_stacked.{ext}")
    plt.close(fig)
est_df = pd.read_csv(B / "benchmark_training_estimates.csv")
if len(est_df):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, r in enumerate(est_df.itertuples()):
        n = np.array([1, 10, 100, 1000]); ax.plot(n, n * r.s_per_iteration / 60, "o-", color=cm(i), label=r.config.replace("_", " "))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("iterations"); ax.set_ylabel("estimated wall time [min] (compute only)"); ax.legend(frameon=False, fontsize=7); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(B / "plots" / f"iteration_time_estimate.{ext}")
    plt.close(fig)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(st[["config", "op", "n", "median", "std", "peak_device_MiB", "throughput"]].round(4).to_string())
print(e2e.round(2).to_string() if len(e2e) else "no e2e")
print(est_df.round(2).to_string()); print(pd.DataFrame(feas)[pd.DataFrame(feas).config.str.contains('native')].round(0).to_string())
