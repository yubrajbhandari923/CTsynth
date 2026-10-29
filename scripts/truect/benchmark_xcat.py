"""Benchmark the XCAT -> mu -> LEAP projection -> LEAP FBP pipeline on real XCAT phantoms (one GPU, serial).

  python benchmark_xcat.py --cases reference copd_22728 copd_21535 --vox 0.8 0.6 --out OUT/xcat_pipeline/benchmarks
  python benchmark_xcat.py --cases ... --grad-checks --vox 2.4        # differentiability checks on a coarse grid

Per (case, voxel size) stages timed with CUDA syncs, `--warmup` warm-ups, `--reps` (7) repetitions; memory in separate profiled runs
(cudaMemGetInfo sampled every 2 ms, a lower bound) so timing runs carry no sampler:
  S1 material table + phantom load to GPU + native-resolution mu conversion      S2 box-average resampling native -> grid (conversion fused)
  A  LEAP forward projection   B LEAP adjoint backprojection   C LEAP helical FBP (GPU tensors)
  D1-D4 leaptorch autograd: forward / backward / forward+loss+backward / +Adam
Full end-to-end runtime comes from run_xcat_pipeline.py metadata (see run_xcat_benchmark.sh). Out-of-memory is recorded, not hidden.
"""
import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/yb107/ctsynth_truect/leap/LEAP/src")
import xcat_lib as X  # noqa: E402
from benchmark_leap import DEV, MiB, MemSampler, dev_used, dot64, sync  # noqa: E402
from leaptorch import FBPFunctionGPU, Projector  # noqa: E402


class Rec:
    def __init__(self, out, tag, reps, warmup):
        self.out, self.tag, self.reps, self.warmup = Path(out), tag, reps, warmup
        self.rows, self.mem, self.oom, self.meta, self.chk = [], [], [], {}, {}
        (self.out / "raw").mkdir(parents=True, exist_ok=True)

    def flush(self):
        for k, v in dict(rows=self.rows, mem=self.mem).items():
            if v:
                with open(self.out / "raw" / f"{self.tag}_{k}.csv", "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(v[0])); w.writeheader(); w.writerows(v)
        self.meta["oom_ops"] = self.oom
        (self.out / "raw" / f"{self.tag}_meta.json").write_text(json.dumps(dict(meta=self.meta, checks=self.chk), indent=1, default=str))

    def record(self, op, fn, prep=None, memruns=2, cold_ok=True):
        try:
            self._record(op, fn, prep, memruns)
        except torch.OutOfMemoryError as e:
            self.oom.append(dict(op=op, error=str(e).split("\n")[0][:300], device_used_MiB=dev_used() / MiB))
            self.rows[:] = [r for r in self.rows if r["op"] != op]
            print(f"[{self.tag}] {op}: CUDA OOM", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        self.flush()

    def _record(self, op, fn, prep, memruns):
        t0 = time.perf_counter(); (prep() if prep else None); sync(); t0 = time.perf_counter(); fn(); sync()
        self.rows.append(dict(config=self.tag, op=op, rep="cold", time_s=time.perf_counter() - t0))
        for _ in range(self.warmup):
            (prep() if prep else None); fn(); sync()
        for r in range(self.reps):
            (prep() if prep else None)
            sync(); t0 = time.perf_counter(); fn(); sync()
            self.rows.append(dict(config=self.tag, op=op, rep=r, time_s=time.perf_counter() - t0))
        pk = dl = tp = 0
        for _ in range(memruns):
            (prep() if prep else None)
            torch.cuda.reset_peak_memory_stats()
            with MemSampler() as ms:
                fn()
            pk, dl, tp = max(pk, ms.peak), max(dl, ms.peak - ms.base), max(tp, torch.cuda.max_memory_allocated())
        self.mem.append(dict(config=self.tag, op=op, peak_device_MiB=pk / MiB, delta_over_baseline_MiB=dl / MiB, torch_alloc_peak_MiB=tp / MiB))
        med = float(np.median([r["time_s"] for r in self.rows if r["op"] == op and r["rep"] != "cold"]))
        print(f"[{self.tag}] {op}: median {med:.4f} s, peak {pk / MiB:.0f} MiB", flush=True)


def setup(case, vox, energy, geometry, margin):
    mm = X.build_material_model(case, energy)
    vol, shape = X.open_phantom(case)
    grid = X.make_grid(shape, X.NATIVE_VOX_MM, vox)
    plan = X.plan_scan(grid, geometry, margin)
    return mm, vol, shape, grid, plan


def bench(case, vox, a):
    tag = f"{case}_{vox:g}mm"
    R = Rec(a.out, tag, a.reps, a.warmup)
    props = torch.cuda.get_device_properties(0)
    mm, vol, shape, grid, plan = setup(case, vox, a.energy_kev, a.geometry, a.margin_mm)
    nvox, nview = int(np.prod(grid.shape)), plan.n_views
    R.meta.update(case=case, vox_mm=vox, grid_zyx=list(grid.shape), n_voxels=nvox, native_shape=list(shape), n_views=nview, n_rotations=plan.n_rotations,
                  proj_shape=[nview, X.N_ROWS, X.N_COLS], n_det_samples=nview * X.N_ROWS * X.N_COLS, scan_length_mm=plan.scan_length_mm,
                  proj_GiB=nview * X.N_ROWS * X.N_COLS * 4 / 2**30, vol_GiB=nvox * 4 / 2**30, gpu=props.name, vram_MiB=props.total_memory / MiB,
                  energy_keV=a.energy_kev, geometry=a.geometry, native_voxels=int(np.prod(shape)), reps=a.reps, warmup=a.warmup)
    print(f"[{tag}] grid {grid.shape}, {nview} views, vol {R.meta['vol_GiB']:.2f} GiB, proj {R.meta['proj_GiB']:.2f} GiB", flush=True)
    R.flush()
    lut = torch.from_numpy(mm.mu_mm).to(DEV)

    def s1():
        mm_ = X.build_material_model(case, a.energy_kev)
        vol_, shp_ = X.open_phantom(case)
        tot = torch.zeros((), device=DEV)
        for z0 in range(0, shp_[0], 64):
            L_ = torch.from_numpy(np.ascontiguousarray(vol_[z0:z0 + 64])).to(DEV)
            tot += lut[L_.long()].sum()
        return tot

    R.record("S1_load_materials_and_native_mu_conversion", s1)
    R.record("S2_resample_to_grid", lambda: X.resample_phantom(vol, grid, mm, DEV))
    lab, grp, rho, mu, hist = X.resample_phantom(vol, grid, mm, DEV)
    del lab, grp, rho
    mu = mu.contiguous()
    P = Projector(forward_project=True, use_static=False, use_gpu=True, gpu_device=DEV, batch_size=1)
    t0 = time.perf_counter(); L = X.make_leap(grid, plan, L=P.leapct); R.meta["setup_geometry_s"] = time.perf_counter() - t0
    gs = torch.zeros((nview, X.N_ROWS, X.N_COLS), device=DEV)
    f = torch.zeros(grid.shape, device=DEV)
    R.record("A_forward_project", lambda: L.project(gs, mu))
    R.chk["A_finite"] = bool(torch.isfinite(gs).all()); R.chk["A_min_max"] = [float(gs.min()), float(gs.max())]
    g = gs.clone()  # simulated projections of the phantom = measured data for B, C and the loss target
    R.record("B_adjoint_backproject", lambda: L.backproject(g, f))
    R.chk["B_finite"] = bool(torch.isfinite(f).all())
    R.record("C_helical_FBP", lambda: L.FBP(g, f))
    R.chk["C_finite"] = bool(torch.isfinite(f).all()); x0 = f.clone()
    R.chk["recon_vs_phantom_mu_rmse_per_mm_whole_grid"] = float(((x0 - mu) ** 2).mean().sqrt())
    del gs, f
    torch.cuda.empty_cache()
    P.vol_data = torch.zeros((1,) + tuple(grid.shape), device=DEV)
    P.proj_data = torch.zeros((1,) + (nview, X.N_ROWS, X.N_COLS), device=DEV)
    xr = x0.clone().unsqueeze(0).requires_grad_(True)
    yt = g.unsqueeze(0)
    lossf = lambda p: 0.5 * ((p - yt) ** 2).mean()  # noqa: E731
    R.record("D1_autograd_forward", lambda: P(xr))
    st = {}

    def prep_bwd():
        xr.grad = None
        st["loss"] = lossf(P(xr))

    R.record("D2_autograd_backward", lambda: st["loss"].backward(), prep=prep_bwd)

    def it():
        xr.grad = None
        lossf(P(xr)).backward()

    R.record("D3_autograd_forward_loss_backward", it)
    opt = torch.optim.Adam([xr], lr=1e-6)

    def it_adam():
        opt.zero_grad(set_to_none=True)
        lossf(P(xr)).backward()
        opt.step()

    R.record("D4_autograd_iteration_plus_Adam", it_adam)
    if xr.grad is not None:
        R.chk["D_grad_finite"] = bool(torch.isfinite(xr.grad).all())
    # operator checks at this size (measured data only, memory-lean): dot product and autograd == adjoint
    del opt, st
    xr = None; P.vol_data = P.proj_data = None; torch.cuda.empty_cache()
    try:
        Ax = L.project(torch.zeros_like(g), mu); Aty = L.backproject(g, torch.zeros(grid.shape, device=DEV))
        a1, b1 = dot64(Ax, g), dot64(mu, Aty)
        R.chk["adjoint_dot_test_phantom_mu_and_simulated_y"] = dict(Ax_dot_y=a1, x_dot_Aty=b1, rel_diff=abs(a1 - b1) / max(abs(a1), abs(b1)))
        del Ax, Aty
        P.vol_data = torch.zeros((1,) + tuple(grid.shape), device=DEV); P.proj_data = torch.zeros((1,) + (nview, X.N_ROWS, X.N_COLS), device=DEV)
        xr2 = x0.clone().unsqueeze(0).requires_grad_(True)
        (P(xr2) * g.unsqueeze(0)).sum().backward()
        gb = L.backproject(g, torch.zeros(grid.shape, device=DEV))
        R.chk["autograd_grad_vs_LEAP_backproject"] = dict(rel_l2=float((xr2.grad[0] - gb).norm() / gb.norm()), max_abs=float((xr2.grad[0] - gb).abs().max()))
    except torch.OutOfMemoryError as e:
        R.chk["checks_oom"] = str(e).split("\n")[0][:200]
    R.flush()
    del P, L, g, x0, mu
    gc.collect(); torch.cuda.empty_cache()


def grad_checks(case, vox, a):
    """Finite-difference gradient checks of the projector and of helical FBP on a coarse grid (float64 residual accumulation)."""
    tag = f"{case}_{vox:g}mm_gradchecks"
    mm, vol, shape, grid, plan = setup(case, vox, a.energy_kev, a.geometry, a.margin_mm)
    chk = dict(case=case, vox_mm=vox, grid_zyx=list(grid.shape), n_views=plan.n_views)
    P = Projector(forward_project=True, use_static=False, use_gpu=True, gpu_device=DEV, batch_size=1)
    L = X.make_leap(grid, plan, L=P.leapct)
    mu = X.resample_phantom(vol, grid, mm, DEV)[3].contiguous()
    vshape, gshape = tuple(grid.shape), (plan.n_views, X.N_ROWS, X.N_COLS)
    y = torch.zeros(gshape, device=DEV); L.project(y, mu)
    x0 = torch.zeros(vshape, device=DEV); L.FBP(y, x0)
    x0 = (x0 + 0.002 * torch.randn(vshape, device=DEV, generator=torch.Generator(device=DEV).manual_seed(1))).contiguous()  # off the zero-residual point
    P.vol_data = torch.zeros((1,) + vshape, device=DEV); P.proj_data = torch.zeros((1,) + gshape, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(0)
    xr = x0.clone().unsqueeze(0).requires_grad_(True)
    (0.5 * ((P(xr) - y.unsqueeze(0)) ** 2).mean()).backward()
    grad = xr.grad[0].clone()

    def Lfun(xx):
        res = (L.project(torch.zeros(gshape, device=DEV), xx) - y).double()
        return float(0.5 * (res * res).mean())

    v = torch.randn(vshape, generator=gen, device=DEV); v = v / v.std() * float(x0.std())
    gd = dot64(grad, v)
    chk["projector_directional_FD"] = {}
    for eps in (1e-3, 1e-2, 1e-1, 1.0):
        fd = (Lfun(x0 + eps * v) - Lfun(x0 - eps * v)) / (2 * eps)
        chk["projector_directional_FD"][str(eps)] = dict(fd=fd, autograd=gd, rel_err=abs(gd - fd) / abs(fd))
    vox_fd = {}
    for i in torch.topk(grad.abs().flatten(), 3).indices.tolist():
        z, yv, xv = np.unravel_index(i, vshape)
        res = {}
        for eps in (1e-2, 1e-1, 1.0):
            xp, xm = x0.clone(), x0.clone(); xp[z, yv, xv] += eps; xm[z, yv, xv] -= eps
            fd = (Lfun(xp) - Lfun(xm)) / (2 * eps)
            res[str(eps)] = dict(fd=fd, autograd=float(grad[z, yv, xv]), rel_err=abs(float(grad[z, yv, xv]) - fd) / abs(fd))
        vox_fd[f"{z},{yv},{xv}"] = res
    chk["projector_single_voxel_FD"] = vox_fd
    # helical FBP as an operator on projections: LEAP FBP_adjoint dot test + leaptorch FBP backward vs finite difference
    del xr, grad, x0, v, P.vol_data, P.proj_data
    P.vol_data = P.proj_data = None
    gc.collect(); torch.cuda.empty_cache()
    w = torch.rand(vshape, generator=gen, device=DEV)
    By = L.FBP(y, torch.zeros(vshape, device=DEV)); Bw = L.FBP_adjoint(torch.zeros(gshape, device=DEV), w)
    a1, b1 = dot64(By, w), dot64(y, Bw)
    chk["FBP_adjoint_dot_test"] = dict(FBPy_dot_w=a1, y_dot_FBPadj_w=b1, rel_diff=abs(a1 - b1) / max(abs(a1), abs(b1)))
    del By, Bw
    torch.cuda.empty_cache()
    y.requires_grad_(True)                                  # y is the leaf: no extra projection-sized copy
    Pv, Pg = torch.zeros((1,) + vshape, device=DEV), torch.zeros((1,) + gshape, device=DEV)
    fo = FBPFunctionGPU.apply(y.unsqueeze(0), Pg, Pv, P.param_id_t)
    (fo * w.unsqueeze(0)).sum().backward()
    gdf_vec = y.grad
    yd = y.detach()
    vv = torch.randn(gshape, generator=gen, device=DEV); vv = vv / vv.std() * float(yd.std())
    gdf = dot64(gdf_vec.view(gshape), vv)
    del Pg, fo, gdf_vec
    y.grad = None
    torch.cuda.empty_cache()
    chk["FBP_autograd_directional_FD"] = {}
    for eps in (1e-3, 1e-2, 1e-1):
        fp = L.FBP(yd + eps * vv, torch.zeros(vshape, device=DEV)); fm = L.FBP(yd - eps * vv, torch.zeros(vshape, device=DEV))
        fd = (dot64(fp, w) - dot64(fm, w)) / (2 * eps)
        chk["FBP_autograd_directional_FD"][str(eps)] = dict(fd=fd, autograd=gdf, rel_err=abs(gdf - fd) / abs(fd))
        del fp, fm
    out = Path(a.out) / "raw"; out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(chk, indent=1, default=str))
    print(f"[{tag}] projector directional FD rel err (eps 1e-2): {chk['projector_directional_FD']['0.01']['rel_err']:.2e}; "
          f"FBP autograd rel err: {chk['FBP_autograd_directional_FD']['0.01']['rel_err']:.2e}", flush=True)
    del P, L
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["reference", "copd_22728", "copd_21535"])
    ap.add_argument("--vox", nargs="+", type=float, default=[0.8, 0.6])
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--energy-kev", type=float, default=70.0)
    ap.add_argument("--geometry", default="truect_helical")
    ap.add_argument("--margin-mm", type=float, default=40.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grad-checks", action="store_true")
    a = ap.parse_args()
    for case in a.cases:
        for v in a.vox:
            (grad_checks if a.grad_checks else bench)(case, v, a)
