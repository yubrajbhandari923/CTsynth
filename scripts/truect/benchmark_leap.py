"""Benchmark LLNL LEAP operators on the TrueCT helical geometry (one SLURM GPU node, one GPU).

Usage: benchmark_leap.py CONFIG [--reps 7] [--warmup 2] [--out DIR] [--no-checks]
  CONFIG: small = 192x192x48 @2.4 mm over the reference scan (3000 views); ref = 768x768x192 @0.6 mm (3000 views);
          copd0 = 768x768x526 @0.6 mm over dcm_000 (9000 views). Detector/scan geometry = recon_final.py (empirical offsets included).
Operations timed (float32, inputs resident on the GPU unless stated; syncs before/after; timed runs carry no memory sampler):
  A project, B backproject (LEAP adjoint of A, unfiltered), C1 FBP (torch GPU tensors), C2 FBP (host numpy arrays, as the pipeline does),
  D1-D4 leaptorch autograd: forward / backward / forward+loss+backward / +Adam step.
Memory: separate profiled runs sample device memory (cudaMemGetInfo, every 2 ms; a lower bound on the true peak) and torch's allocator peak.
Checks: dot-product adjoint tests, autograd-vs-backproject, finite-difference gradient checks, FBP autograd test (small config only for FD).
Writes raw per-config files under OUT/raw/; benchmark_aggregate.py builds the CSV/summary/plots. Reads the dataset, never writes to it.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/yb107/ctsynth_truect/leap/LEAP/src")
T_IMPORT0 = time.perf_counter()
from leapctype import tomographicModels  # noqa: E402,F401
from leaptorch import Projector, ProjectorFunctionGPU, FBPFunctionGPU  # noqa: E402

from common import OUT  # noqa: E402
from leap_recon import SOD, SDD, PIX_MM, load_g  # noqa: E402

T_IMPORT = time.perf_counter() - T_IMPORT0
MiB = 2.0**20
DEV = torch.device("cuda:0")
# recon_final.py settings (rot_sign -1, rows flipped, channel shift -1, volume offset): the existing pipeline's empirical conventions
PIPE = dict(rot_sign=-1, ch_flip=False, row_flip=True, col_shift=-1.0, off_x_mm=-0.1, off_y_mm=-0.3)


def sync():
    torch.cuda.synchronize()


def dev_used():
    free, total = torch.cuda.mem_get_info()
    return total - free


class MemSampler:
    def __init__(self, dt=0.002):
        self.dt, self.peak, self._stop = dt, 0, threading.Event()

    def __enter__(self):
        sync()
        self.base = self.peak = dev_used()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, dev_used())
            time.sleep(self.dt)

    def __exit__(self, *a):
        sync()
        self.peak = max(self.peak, dev_used())
        self._stop.set()
        self.t.join()


def cfg_of(name):
    rj = lambda s: json.load(open(OUT / "reports" / f"recon_{s}.json"))
    ref, c0 = rj("reference_leap_recon"), rj("copd_dcm_000_recon")
    if name == "ref":
        return dict(series="reference", nxy=ref["nxy"], vox=ref["vox_mm"], nz=ref["nz"], dz=ref["dz_mm"])
    if name == "copd0":
        return dict(series="dcm_000", nxy=c0["nxy"], vox=c0["vox_mm"], nz=c0["nz"], dz=c0["dz_mm"])
    if name == "small":
        f = 4  # same FOV/z-extent as ref, 4x coarser voxels
        return dict(series="reference", nxy=ref["nxy"] // f, vox=ref["vox_mm"] * f, nz=ref["nz"] // f, dz=ref["dz_mm"] * f)
    raise SystemExit("config must be small|ref|copd0")


def dot64(a, b):
    return float(torch.sum(a * b, dtype=torch.float64))


def run(name, reps, warmup, out, checks, only_checks=False):
    out = Path(out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cfg = cfg_of(name)
    rows, mem, setup, meta, chk = [], [], [], {}, {}

    def add_setup(stage, t, note=""):
        setup.append(dict(config=name, stage=stage, time_s=t, note=note))

    add_setup("python_import_leapctype_leaptorch_torch", T_IMPORT, "after torch import")
    t0 = time.perf_counter(); torch.zeros(1, device=DEV); sync(); add_setup("cuda_context_init", time.perf_counter() - t0)
    ctx_used = dev_used()
    # ---- data (disk I/O, conversion) ----
    t0 = time.perf_counter()
    g_np, ang, zmm = load_g(cfg["series"], 0, None, PIPE["ch_flip"], PIPE["row_flip"])
    add_setup("disk_read_and_convert_projections", time.perf_counter() - t0, f"{g_np.nbytes / 1e9:.2f} GB float32 host")
    n, rows_d, cols_d = g_np.shape
    pitch = float(np.polyfit(ang, zmm, 1)[0])
    # ---- geometry / operator setup ----
    t0 = time.perf_counter()
    P = Projector(forward_project=True, use_static=False, use_gpu=True, gpu_device=DEV, batch_size=1)
    add_setup("leaptorch_Projector_construct", time.perf_counter() - t0)
    L = P.leapct
    sg = PIPE["rot_sign"]
    phis = np.degrees(sg * ang).astype(np.float32)
    t0 = time.perf_counter()
    L.set_conebeam(n, rows_d, cols_d, PIX_MM, PIX_MM, 0.5 * (rows_d - 1), 0.5 * (cols_d - 1) + PIPE["col_shift"], phis, SOD, SDD)
    L.set_curvedDetector()
    L.set_helicalPitch(sg * pitch)
    zmid = 0.5 * (zmm[0] + zmm[-1])
    nz, nxy = cfg["nz"], cfg["nxy"]
    L.set_volume(nxy, nxy, nz, cfg["vox"], cfg["dz"], PIPE["off_x_mm"], PIPE["off_y_mm"], 0.0)
    P.vol_data = P.proj_data = None  # drop leaptorch's own zero buffers; we allocate explicitly below
    add_setup("geometry_set_conebeam_curved_helical_volume", time.perf_counter() - t0)
    vshape, gshape = (nz, nxy, nxy), (n, rows_d, cols_d)
    meta.update(config=name, cfg=cfg, n_views=n, det_rows=rows_d, det_cols=cols_d, vol_shape_zyx=list(vshape), proj_shape=list(gshape),
                pitch_mm_per_rad=pitch, z_scan_mm=[float(zmm[0]), float(zmm[-1])], pipeline_conventions=PIPE, sod=SOD, sdd=SDD,
                n_det_samples=n * rows_d * cols_d, n_voxels=nz * nxy * nxy, ctx_used_MiB=ctx_used / MiB,
                proj_GiB=n * rows_d * cols_d * 4 / 2**30, vol_GiB=nz * nxy * nxy * 4 / 2**30)
    # ---- device tensors ----
    t0 = time.perf_counter(); g = torch.from_numpy(g_np).to(DEV); sync(); add_setup("H2D_projections", time.perf_counter() - t0)
    f = torch.zeros(vshape, device=DEV)
    gs = torch.zeros(gshape, device=DEV)
    sync()

    oom_ops = []

    def flush():
        import csv
        if only_checks:
            return
        for key, v in dict(rows=rows, mem=mem, setup=setup).items():
            if v:
                with open(out / "raw" / f"{key}_{name}.csv", "w", newline="") as fh:
                    w_ = csv.DictWriter(fh, fieldnames=list(v[0].keys())); w_.writeheader(); w_.writerows(v)
        chk["oom_ops"] = oom_ops
        json.dump(dict(meta=meta, checks=chk), open(out / "raw" / f"meta_checks_{name}.json", "w"), indent=1, default=str)

    def record(op, mode, fn, prep=None, memruns=2):
        if only_checks:
            return
        try:
            _record(op, mode, fn, prep, memruns)
        except torch.OutOfMemoryError as e:
            oom_ops.append(dict(op=op, error=str(e).split("\n")[0][:300], device_used_MiB=dev_used() / MiB))
            print(f"[{name}] {op}: CUDA OOM ({oom_ops[-1]['error'][:120]})", flush=True)
            rows[:] = [r_ for r_ in rows if r_["op"] != op]
            import gc; gc.collect(); torch.cuda.empty_cache()
        flush()

    def _record(op, mode, fn, prep, memruns):
        fn_ = lambda: (prep() if prep else None, fn())  # noqa: E731  (cold call only)
        # cold first call
        sync(); t0 = time.perf_counter(); fn_(); sync()
        rows.append(dict(config=name, op=op, mode=mode, rep="cold", time_s=time.perf_counter() - t0))
        for _ in range(warmup):
            fn_(); sync()
        for r in range(reps):
            if prep:
                prep()
            sync(); t0 = time.perf_counter(); fn(); sync()
            rows.append(dict(config=name, op=op, mode=mode, rep=r, time_s=time.perf_counter() - t0))
        pk, dl, tp = 0, 0, 0
        for _ in range(memruns):
            if prep:
                prep()
            torch.cuda.reset_peak_memory_stats()
            with MemSampler() as ms:
                fn()
            pk, dl, tp = max(pk, ms.peak), max(dl, ms.peak - ms.base), max(tp, torch.cuda.max_memory_allocated())
        mem.append(dict(config=name, op=op, mode=mode, peak_device_MiB=pk / MiB, delta_over_baseline_MiB=dl / MiB, torch_alloc_peak_MiB=tp / MiB))
        print(f"[{name}] {op}: median {np.median([r_['time_s'] for r_ in rows if r_['op'] == op and r_['rep'] != 'cold']):.4f} s, peak {pk / MiB:.0f} MiB", flush=True)

    # ===== C (first: it creates the realistic volume x used by A and D) =====
    t0 = time.perf_counter(); L.FBP(g, f); sync(); add_setup("initial_FBP_for_realistic_x_cold", time.perf_counter() - t0)
    chk["fbp_torch_finite"] = bool(torch.isfinite(f).all()); chk["fbp_shape"] = list(f.shape)
    chk["fbp_mu_stats_1_per_mm"] = dict(mean=float(f.mean()), min=float(f.min()), max=float(f.max()), p99=float(torch.quantile(f.flatten()[::97], 0.99)))
    x = f.clone()
    record("C1_FBP_helical_gpu_tensors", "torch cuda tensors; LEAP copies g (inplace=False)", lambda: L.FBP(g, f))
    # C2 host path: numpy in/out as in recon_final.py. Detect in-place modification of g by LEAP and restore it untimed.
    g0 = g_np.copy(); f_np = np.zeros(vshape, np.float32)
    probe = g_np[::50].copy()
    L.FBP(g_np, f_np)
    mut = not np.array_equal(probe, g_np[::50])
    meta["host_fbp_modifies_g_in_place"] = bool(mut)
    chk["fbp_host_finite"] = bool(np.isfinite(f_np).all())
    chk["fbp_host_vs_gpu_tensors_max_abs_diff"] = float(np.abs(f_np - f.cpu().numpy()).max())
    record("C2_FBP_helical_host_numpy", "numpy in/out (H2D/D2H inside LEAP), pipeline path", lambda: L.FBP(g_np, f_np),
           prep=(lambda: np.copyto(g_np, g0)) if mut else None, memruns=1)
    del g0, f_np
    # ===== A, B =====
    record("A_forward_project", "torch cuda tensors, LEAP project", lambda: L.project(gs, x))
    chk["A_finite"] = bool(torch.isfinite(gs).all()); chk["A_shape"] = list(gs.shape)
    chk["A_line_integral_stats"] = dict(min=float(gs.min()), max=float(gs.max()), mean=float(gs.mean()))
    record("B_adjoint_backproject", "torch cuda tensors, LEAP backproject (A^T, no filter)", lambda: L.backproject(g, f))
    chk["B_finite"] = bool(torch.isfinite(f).all()); chk["B_shape"] = list(f.shape)
    # ===== D: leaptorch autograd =====
    del gs, f  # A/B/C scratch buffers are not needed during autograd; frees VRAM on the 24 GB card (same for every config)
    torch.cuda.empty_cache()
    P.vol_data = torch.zeros((1,) + vshape, device=DEV)
    P.proj_data = torch.zeros((1,) + gshape, device=DEV)
    xr = x.clone().unsqueeze(0).requires_grad_(True)
    yt = g.unsqueeze(0)
    lossf = lambda p: 0.5 * ((p - yt) ** 2).mean()  # noqa: E731
    record("D1_torch_forward", "leaptorch Projector.forward (autograd graph built)", lambda: P(xr))
    st = {}

    def prep_bwd():
        xr.grad = None
        st["loss"] = lossf(P(xr))

    record("D2_torch_backward", "loss.backward() through projector (graph built untimed)", lambda: st["loss"].backward(), prep=prep_bwd)

    def it():
        xr.grad = None
        lossf(P(xr)).backward()

    record("D3_torch_forward_loss_backward", "one full autograd iteration", it)
    opt = torch.optim.Adam([xr], lr=1e-6)

    def it_adam():
        opt.zero_grad(set_to_none=True)
        lossf(P(xr)).backward()
        opt.step()

    record("D4_torch_iteration_plus_Adam", "forward+loss+backward+Adam.step", it_adam)
    if xr.grad is not None:
        chk["D_grad_finite"] = bool(torch.isfinite(xr.grad).all()); chk["D_grad_shape"] = list(xr.grad.shape)
    # step-wise timing of a training iteration is not split further; see D1/D2/D3.
    t0 = time.perf_counter(); _ = xr.detach().cpu().numpy(); add_setup("D2H_volume_result", time.perf_counter() - t0, f"{xr.numel() * 4 / 1e9:.2f} GB")

    # ===== validation =====
    if checks:
        lowmem = name == "copd0"  # 9000 views: avoid extra random 2 GB tensors on the 24 GB card; use measured data only
        keys_before = set(chk)
        del opt, st
        xr = None; P.vol_data = P.proj_data = None; torch.cuda.empty_cache()  # LEAP allocates 3D textures: leave it room
        gen = torch.Generator(device=DEV).manual_seed(0)
        rnd = lambda shape: torch.rand(shape, generator=gen, device=DEV)  # noqa: E731
        # dot-product adjoint tests <Ax,y> vs <x,A^T y>
        dots = {}
        pairs = [("recon_x_measured_y", x, g)] if lowmem else [("random_x_random_y", rnd(vshape), rnd(gshape)), ("recon_x_measured_y", x, g)]
        for tag, xx, yy in pairs:
            Ax = L.project(torch.zeros(gshape, device=DEV), xx); Aty = L.backproject(yy, torch.zeros(vshape, device=DEV))
            a, b = dot64(Ax, yy), dot64(xx, Aty)
            del Ax, Aty; torch.cuda.empty_cache()
            dots[tag] = dict(Ax_dot_y=a, x_dot_Aty=b, rel_diff=abs(a - b) / max(abs(a), abs(b)))
        chk["adjoint_dot_test_A_vs_backproject"] = dots
        P.vol_data = torch.zeros((1,) + vshape, device=DEV); P.proj_data = torch.zeros((1,) + gshape, device=DEV)
        xr = x.clone().unsqueeze(0).requires_grad_(True)
        # autograd backward vs explicit backproject
        r = g if lowmem else rnd(gshape)
        xr.grad = None
        (P(xr) * r.unsqueeze(0)).sum().backward()
        ga = xr.grad[0].clone()
        gb = L.backproject(r, torch.zeros(vshape, device=DEV))
        chk["autograd_grad_vs_LEAP_backproject"] = dict(r="measured g" if lowmem else "random", rel_l2=float((ga - gb).norm() / gb.norm()), max_abs=float((ga - gb).abs().max()),
                                                          note="autograd backward of Projector calls LEAP backproject on grad_output")
        chk["leaptorch_aliasing_note"] = "Projector.forward returns/overwrites a shared proj_data buffer and backward returns the shared vol_data buffer; clone grads before the next call"
        # FBP operator: LEAP's FBP_adjoint is documented as NOT an exact adjoint for helical scans
        if name == "small":
            w = rnd(vshape); y = rnd(gshape) * 0.0 + g
            By = L.FBP(g, torch.zeros(vshape, device=DEV)); Bw = L.FBP_adjoint(torch.zeros(gshape, device=DEV), w)
            a, b = dot64(By, w), dot64(g, Bw)
            chk["FBP_adjoint_dot_test"] = dict(FBPy_dot_w=a, y_dot_FBPadj_w=b, rel_diff=abs(a - b) / max(abs(a), abs(b)),
                                              note="LEAP docs: FBP_adjoint is not an exact adjoint for helical scans")
            # FBP autograd: leaptorch FBPFunctionGPU.backward vs finite difference of L(y)=<w,FBP(y)> (linear in y)
            gi = g.clone().unsqueeze(0).requires_grad_(True)
            Pv = torch.zeros((1,) + vshape, device=DEV); Pg = torch.zeros((1,) + gshape, device=DEV)
            pidt = P.param_id_t
            fo = FBPFunctionGPU.apply(gi, Pg, Pv, pidt)
            (fo * w.unsqueeze(0)).sum().backward()
            grad_fbp = gi.grad[0].clone()
            v = torch.randn(gshape, generator=gen, device=DEV)
            v = v / v.std() * g.std()
            gd = dot64(grad_fbp, v)
            fd = {}
            for eps in (1e-3, 1e-2, 1e-1):
                with torch.no_grad():
                    fp = L.FBP((g + eps * v).clone(), torch.zeros(vshape, device=DEV)); fm = L.FBP((g - eps * v).clone(), torch.zeros(vshape, device=DEV))
                fdv = (dot64(fp, w) - dot64(fm, w)) / (2 * eps)
                fd[str(eps)] = dict(fd=fdv, autograd=gd, rel_err=abs(gd - fdv) / abs(fdv))
            chk["FBP_autograd_directional_FD"] = fd
            chk["FBP_autograd_note"] = "backward uses lct.fbp_adjoint_gpu; leaptorch source comments it 'needs to be replaced'; relative errors here are the evidence, not the claim"
        # finite-difference gradient check for the differentiable forward projector (small config only: float64 residual copies)
        if name == "small":
            yy = g
            def Lfun(xx):
                res = (L.project(torch.zeros(gshape, device=DEV), xx) - yy).double()
                return float(0.5 * (res * res).mean())
            xr = x.clone().unsqueeze(0).requires_grad_(True)
            P.vol_data.zero_()
            loss = 0.5 * ((P(xr) - yy.unsqueeze(0)) ** 2).mean()
            xr.grad = None; loss.backward()
            grad = xr.grad[0].clone()
            v = torch.randn(vshape, generator=gen, device=DEV); v = v / v.std() * float(x.std())
            gd = dot64(grad, v)
            fdres = {}
            for eps in (1e-3, 1e-2, 1e-1, 1.0):
                fdv = (Lfun(x + eps * v) - Lfun(x - eps * v)) / (2 * eps)
                fdres[str(eps)] = dict(fd=fdv, autograd=gd, rel_err=abs(gd - fdv) / abs(fdv))
            chk["projector_directional_FD"] = dict(loss="0.5*mean((Ax-y)^2)", direction="gaussian, rms = rms(x)", results=fdres)
            # single-voxel checks at the 3 largest-|grad| in-FOV voxels
            idx = torch.topk(grad.abs().flatten(), 3).indices.tolist()
            vox = {}
            for i in idx:
                z, yv, xv = np.unravel_index(i, vshape)
                res = {}
                for eps in (1e-2, 1e-1, 1.0):
                    xp, xm = x.clone(), x.clone(); xp[z, yv, xv] += eps; xm[z, yv, xv] -= eps
                    fdv = (Lfun(xp) - Lfun(xm)) / (2 * eps)
                    res[str(eps)] = dict(fd=fdv, autograd=float(grad[z, yv, xv]), rel_err=abs(float(grad[z, yv, xv]) - fdv) / abs(fdv))
                vox[f"{z},{yv},{xv}"] = res
            chk["projector_single_voxel_FD"] = vox
        validation_keys = [k for k in chk if k not in keys_before]

    if only_checks:
        json.dump({k: chk[k] for k in validation_keys}, open(out / "raw" / f"checks_only_{name}.json", "w"), indent=1, default=str)
    flush()


def env_info():
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()  # noqa: E731
    leap = "/home/yb107/ctsynth_truect/leap/LEAP"
    return dict(
        gpu=sh("nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit --format=csv,noheader"),
        cpu=sh("lscpu | grep -E 'Model name|^CPU\\(s\\)|Socket' | tr '\\n' ';'"), mem_total=sh("grep MemTotal /proc/meminfo"),
        node=os.environ.get("SLURMD_NODENAME"), slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_cpus=os.environ.get("SLURM_CPUS_PER_TASK"),
        slurm_mem=os.environ.get("SLURM_MEM_PER_NODE"), loaded_modules=os.environ.get("LOADEDMODULES"),
        nvcc=sh("nvcc --version | tail -2 | tr '\\n' ' '"), torch=torch.__version__, torch_cuda=torch.version.cuda,
        torch_cudnn=str(torch.backends.cudnn.version()), python=sys.version.split()[0], numpy=np.__version__,
        leap_commit=sh(f"git -C {leap} rev-parse --short HEAD"), leap_lib=f"{leap}/build/lib/libleapct.so",
        leap_lib_mtime=sh(f"stat -c %y {leap}/build/lib/libleapct.so"), leap_build=sh("cat /home/yb107/ctsynth_truect/build_leap.sh"),
        benchmark_script_sha=sh(f"sha1sum {__file__}"), argv=sys.argv, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=str(OUT / "benchmarks"))
    ap.add_argument("--no-checks", action="store_true")
    ap.add_argument("--only-checks", action="store_true", help="skip timing; rerun validation only and write raw/checks_only_<config>.json (merged by benchmark_aggregate.py)")
    a = ap.parse_args()
    Path(a.out, "raw").mkdir(parents=True, exist_ok=True)
    if not a.only_checks:
        json.dump(env_info(), open(Path(a.out) / "raw" / f"env_{a.config}.json", "w"), indent=1)
    run(a.config, a.reps, a.warmup, a.out, not a.no_checks, a.only_checks)
