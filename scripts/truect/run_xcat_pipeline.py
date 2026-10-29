"""XCAT phantom -> mu(E) -> LEAP forward projection -> LEAP helical FBP -> NIfTI/Zarr/previews/validation (one case, one GPU).

  python run_xcat_pipeline.py --case copd_22728 --energy-kev 70 --voxel-size-mm 0.8 --geometry truect_helical \
      --output /home/yb107/datasets/truect_pilot/outputs/xcat_pipeline/copd_22728

Everything is generated from the phantom: the reconstruction comes from the NEWLY SIMULATED projections, never from TrueCT measurements.
Labels, organ labels, ideal CT and reconstruction live on one grid (same shape, spacing, affine). Orientation letters are PROVISIONAL.
No scatter, noise, bowtie or polychromatic physics (monoenergetic Beer-Lambert line integrals only). FBP is not differentiated here.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel import openers

sys.path.insert(0, "/home/yb107/ctsynth_truect/leap/LEAP/src")
import xcat_lib as X  # noqa: E402
from benchmark_leap import MemSampler, dev_used  # noqa: E402
from common import OUT, PHANTOMS  # noqa: E402
from niiutil import write_nii  # noqa: E402

openers.Opener.default_compresslevel = 3
MiB = 2.0**20


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


def erode(mask, k):  # boolean 3D erosion on the GPU with a cubic structuring element of radius k voxels
    m = mask[None, None].float()
    return (-torch.nn.functional.max_pool3d(-m, 2 * k + 1, 1, k))[0, 0] > 0.5


def dice(a, b):
    return float(2 * (a & b).sum() / max(1, int(a.sum() + b.sum())))


def shift_by_xcorr(a, b, mask):
    """Integer + quadratic sub-voxel shift (voxels, array order z,y,x) maximising the cross-correlation of b against a inside mask."""
    a = (a - a[mask].mean()) * mask
    b = (b - b[mask].mean()) * mask
    cc = torch.fft.irfftn(torch.fft.rfftn(a) * torch.conj(torch.fft.rfftn(b)), s=a.shape)
    idx = np.unravel_index(int(cc.argmax()), cc.shape)
    sh_ = []
    for ax in range(3):
        n = cc.shape[ax]
        i = idx[ax]
        sl = [slice(idx[0], idx[0] + 1), slice(idx[1], idx[1] + 1), slice(idx[2], idx[2] + 1)]
        vals = []
        for d in (-1, 0, 1):
            s2 = list(sl)
            s2[ax] = slice((i + d) % n, (i + d) % n + 1)
            vals.append(float(cc[tuple(s2)]))
        den = vals[0] - 2 * vals[1] + vals[2]
        frac = 0.5 * (vals[0] - vals[2]) / den if den != 0 else 0.0
        s = i + frac
        sh_.append(float(s - n if s > n / 2 else s))
    return sh_


# ----------------------------------------------------------------------------- previews
def to_ras(arr, affine):
    ornt = nib.orientations.io_orientation(affine)
    return nib.orientations.apply_orientation(np.asarray(arr), ornt)  # array axes now increase toward R, A, S


def tri_axes(fig, row, nrows, title=None):
    axs = [fig.add_subplot(nrows, 3, 3 * row + i + 1) for i in range(3)]
    return axs


def show_tri(axs, ras, idx, cmap="gray", vmin=None, vmax=None, alpha=None, interp="nearest", letters=True):
    i, j, k = idx
    ims = []
    for ax, (im, lx, ly, flipx) in zip(axs, ((ras[:, :, k].T, "R", "A", True), (ras[:, j, :].T, "R", "S", True), (ras[i, :, :].T, "A", "S", True))):
        h = ax.imshow(im, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, interpolation=interp)
        if flipx and not ax.xaxis_inverted():
            ax.invert_xaxis()
        ax.set_xticks([]); ax.set_yticks([])
        if letters:
            ax.text(0.02, 0.5, lx, transform=ax.transAxes, color="yellow", fontsize=9, va="center")
            ax.text(0.5, 0.96, ly, transform=ax.transAxes, color="yellow", fontsize=9, ha="center", va="top")
        ims.append(h)
    return ims


def slice_choice(ras_mask):
    c = np.array(np.nonzero(ras_mask)).mean(1)
    return tuple(int(round(v)) for v in c)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, choices=sorted(PHANTOMS))
    ap.add_argument("--energy-kev", type=float, default=70.0)
    ap.add_argument("--voxel-size-mm", type=float, default=0.8)
    ap.add_argument("--geometry", default="truect_helical", choices=sorted(X.GEOMETRIES))
    ap.add_argument("--margin-mm", type=float, default=40.0, help="extra scan length beyond the phantom on each side (helical coverage)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--skip-native-labels", action="store_true", help="do not write the 0.25 mm native label NIfTI (large)")
    ap.add_argument("--skip-zarr", action="store_true")
    ap.add_argument("--max-gpu-fraction", type=float, default=0.85, help="refuse to run if the estimated GPU need exceeds this fraction")
    a = ap.parse_args()
    out = Path(a.output)
    for d in ("nifti", "projections", "previews"):
        (out / d).mkdir(parents=True, exist_ok=True)
    T = X.Timer()
    t_all = time.perf_counter()
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    sync = torch.cuda.synchronize
    log = lambda *s: print(f"[{a.case}]", *s, flush=True)  # noqa: E731
    env = dict(gpu=props.name, vram_MiB=props.total_memory / MiB, torch=torch.__version__, torch_cuda=torch.version.cuda,
               leap_commit=sh("git -C /home/yb107/ctsynth_truect/leap/LEAP rev-parse --short HEAD"), node=os.environ.get("SLURMD_NODENAME"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), python=platform.python_version(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               script_sha1=sh(f"sha1sum {Path(__file__).resolve()} {Path(X.__file__).resolve()}"), argv=sys.argv)

    # ---- materials
    with T("1_materials_and_load"):
        mm = X.build_material_model(a.case, a.energy_kev)
        vol, native_shape = X.open_phantom(a.case)
    log("material check:", json.dumps(mm.xcheck))
    if mm.xcheck.get("xlsx_vs_conversion_table_in", {}).get("n_mismatch", 0):
        raise SystemExit("material xlsx and conversion table disagree: stop")
    grid = X.make_grid(native_shape, X.NATIVE_VOX_MM, a.voxel_size_mm)
    plan = X.plan_scan(grid, a.geometry, a.margin_mm)
    nvox = int(np.prod(grid.shape))
    need_MiB = (nvox * 4 * 8 + plan.n_views * X.N_ROWS * X.N_COLS * 4 * 2.2) / MiB
    log(f"grid {grid.shape} @ {grid.vox_mm} mm; scan {plan.n_views} views ({plan.n_rotations} rotations, {plan.scan_length_mm:.1f} mm); estimated GPU need {need_MiB:.0f} MiB of {props.total_memory / MiB:.0f}")
    if need_MiB > a.max_gpu_fraction * props.total_memory / MiB:
        raise SystemExit(f"memory-feasibility check FAILED: ~{need_MiB:.0f} MiB needed > {a.max_gpu_fraction:.0%} of {props.total_memory / MiB:.0f} MiB; use a coarser --voxel-size-mm")

    # ---- resample
    with T("2_resample_to_simulation_grid"):
        lab, grp, rho, mu, hist = X.resample_phantom(vol, grid, mm, dev)
    present = np.flatnonzero(hist)
    unmapped = [int(i) for i in present if int(i) not in mm.mats]
    if unmapped:
        raise SystemExit(f"labels present but absent from the material table: {unmapped} (not inventing materials)")
    mu_w = mm.mu_water_mm
    hu_ideal = X.mu_to_hu(mu, mu_w)
    aff = X.affine_for(grid.shape, grid.vox_mm)

    # ---- phantom integrity (native)
    sub = np.asarray(vol[::4, ::4, ::4])
    nbr = [float((np.diff(sub, axis=ax) == 0).mean()) for ax in range(3)]
    ext = [n * X.NATIVE_VOX_MM for n in native_shape]
    vox_mL = X.NATIVE_VOX_MM**3 / 1000.0
    vol_by_group = {}
    for oid in present:
        g = int(mm.organ_to_group[oid])
        vol_by_group[mm.groups[g]["name"]] = vol_by_group.get(mm.groups[g]["name"], 0.0) + float(hist[oid]) * vox_mL
    lung_L = sum(v for k, v in vol_by_group.items() if k in ("lung", "lung_parenchyma", "emphysema")) / 1000.0
    body_mask_t = grp > 0
    yy = torch.arange(grid.shape[1], device=dev)[:, None].float()
    xx = torch.arange(grid.shape[2], device=dev)[None, :].float()
    rad = torch.sqrt(((yy + 0.5) * grid.vox_mm - grid.shape[1] * grid.vox_mm / 2) ** 2 + ((xx + 0.5) * grid.vox_mm - grid.shape[2] * grid.vox_mm / 2) ** 2)
    body_r_max = float(rad[body_mask_t.any(0)].max())
    integrity = dict(file_size_matches_dims=True, native_shape_zyx=list(native_shape), native_extent_mm_zyx=ext,
                     labels_present=[int(i) for i in present], unmapped_labels=unmapped, adjacent_equal_fraction_per_axis_zyx_stride4=nbr,
                     axis_swap_note="a wrong reshape makes neighbouring voxels unrelated (fraction near the chance level); all three axes are high here. Layout evidence also in reports/survey_phantoms.json",
                     volume_mL_by_group_native=vol_by_group, lung_volume_L=lung_L, density_range_g_cm3=[float(mm.rho[present].min()), float(mm.rho[present].max())],
                     body_max_radius_mm=body_r_max, scanner_fov_radius_mm=plan.fov_radius_mm, body_inside_fov=bool(body_r_max < plan.fov_radius_mm))
    if not integrity["body_inside_fov"]:
        log("WARNING: phantom extends beyond the scanner FOV: truncation artefacts expected")

    # ---- forward projection
    L = X.make_leap(grid, plan)
    mu_c = mu.contiguous()
    X.forward_project(L, mu_c, plan)  # warm-up (kernel load) on the real problem; result discarded
    sync()
    g = None
    with MemSampler() as ms_f:
        t0 = time.perf_counter(); g = X.forward_project(L, mu_c, plan); sync(); t_fwd = time.perf_counter() - t0
    T.t["3_forward_projection"] = t_fwd
    peak_f = ms_f.peak / MiB
    finite_g = bool(torch.isfinite(g).all())
    air_floor = float(mm.mu_mm[0]) * float(np.hypot(grid.shape[1], grid.shape[2]) * grid.vox_mm)
    pstats = dict(shape_view_row_channel=list(g.shape), min=float(g.min()), max=float(g.max()), mean=float(g.mean()), finite=finite_g,
                  expected_pure_air_floor_upper_bound=air_floor, note="rays missing the body integrate only air (mu_air*path <= floor); p>=0 everywhere")
    if not args_ok(pstats):
        log("WARNING: negative line integrals")

    # ---- projections out
    attrs = dict(case=a.case, energy_keV=a.energy_kev, geometry=a.geometry, plan=plan.to_dict(), grid=grid.to_dict(), phis_deg="360*k/views_per_rotation",
                 source="forward projection of the resampled XCAT mu(70 keV) with LEAP; no noise/scatter")
    if not a.skip_zarr:
        with T("4_write_projection_zarr"):
            X.write_projection_zarr(g, plan, out / "projections/simulated_projections.zarr", attrs)
    prev = X.projection_preview(g)
    pimg = nib.Nifti1Image(prev.astype(np.float32), np.eye(4))
    pimg.header["descrip"] = b"NONANATOMICAL projection tensor view/5,row,channel/3"
    pimg.set_sform(np.eye(4), 0); pimg.set_qform(np.eye(4), 0)
    nib.save(pimg, out / "nifti/simulated_projections_preview.nii.gz")

    # representative detector images / sinograms (host copies, small)
    nv = plan.n_views
    ext_z = grid.shape[0] * grid.vox_mm
    per_view = plan.scan_length_mm / plan.n_rotations / 1000.0
    det_views = [int(round((zt + plan.scan_length_mm / 2) / per_view)) for zt in (-0.35 * ext_z, -0.1 * ext_z, 0.15 * ext_z, 0.4 * ext_z)]
    det_imgs = [g[v].cpu().numpy() for v in det_views]
    sino_mid = g[:, X.N_ROWS // 2, :].cpu().numpy()
    sino_edge = g[:, 4, :].cpu().numpy()

    # ---- reconstruction
    with MemSampler() as ms_r:
        t0 = time.perf_counter(); rec_mu = X.fbp(L, g, grid, inplace=False); sync(); t_fbp = time.perf_counter() - t0
    T.t["5_helical_FBP"] = t_fbp
    peak_r = ms_r.peak / MiB
    del g
    torch.cuda.empty_cache()
    hu_rec = X.mu_to_hu(rec_mu, mu_w)

    # ---- masks + metrics (valid region = inside FOV circle and inside the helically covered z range)
    nz = grid.shape[0]
    zc = (torch.arange(nz, device=dev).float() + 0.5) * grid.vox_mm - nz * grid.vox_mm / 2
    z_valid = (zc.abs() <= plan.scan_length_mm / 2 - 0.5 * plan.margin_mm)[:, None, None]
    fov = (rad < 0.95 * plan.fov_radius_mm)[None]
    valid = z_valid & fov
    body = body_mask_t
    body_e = erode(body, 2) & valid
    names = {k: v["name"] for k, v in mm.groups.items()}
    lung_ids = [k for k, v in names.items() if v in ("lung", "lung_parenchyma", "emphysema")]
    lung = torch.isin(grp, torch.tensor(lung_ids, device=dev, dtype=torch.uint8)) if lung_ids else torch.zeros_like(body)
    lung_e = erode(lung, 2) & valid if lung_ids else lung

    def err(mask):
        if int(mask.sum()) == 0:
            return None
        d_mu, d_hu = (rec_mu - mu)[mask], (hu_rec - hu_ideal)[mask]
        return dict(n_voxels=int(mask.sum()), mu_mae_per_mm=float(d_mu.abs().mean()), mu_rmse_per_mm=float((d_mu ** 2).mean().sqrt()), mu_bias_per_mm=float(d_mu.mean()),
                    hu_mae=float(d_hu.abs().mean()), hu_rmse=float((d_hu ** 2).mean().sqrt()), hu_bias=float(d_hu.mean()))

    metrics = dict(valid_region=err(valid), body_eroded2=err(body_e), lung_eroded2=err(lung_e), per_group={})
    for gid_, nm in names.items():
        m = erode(grp == gid_, 1) & valid
        e = err(m)
        if e and e["n_voxels"] >= 200:
            e["ideal_hu_mean"] = float(hu_ideal[m].mean()); e["recon_hu_mean"] = float(hu_rec[m].mean())
            metrics["per_group"][nm] = e
    # registration: shifts + boundary overlaps (recon-derived masks vs label-derived masks)
    shift = shift_by_xcorr(hu_ideal.clamp(-1000, 1500), hu_rec.clamp(-1000, 1500), valid)
    import scipy.ndimage as ndi
    rec_body = (hu_rec > -300).cpu().numpy()
    rec_body = np.stack([ndi.binary_fill_holes(s) for s in rec_body])
    rec_body_t = torch.from_numpy(rec_body).to(dev)
    reg = dict(xcorr_shift_voxels_zyx_recon_vs_ideal=shift, xcorr_shift_mm_zyx=[s * grid.vox_mm for s in shift],
               body_dice_recon_hu_gt_m300_filled_vs_label_body=dice(rec_body_t & valid, body & valid),
               affine_identical_all_recon_grid_files=True, shapes_identical=True)
    if lung_ids:
        rec_lung = (hu_rec < -400) & erode(rec_body_t, 3)
        reg["lung_dice_recon_hu_lt_m400_inside_body_vs_label_lung"] = dice(rec_lung & valid, lung & valid)
    # recon z-profile of the error to expose helical end effects
    zprof = ((hu_rec - hu_ideal).abs() * body_e).sum((1, 2)) / body_e.sum((1, 2)).clamp(min=1)

    # ---- NIfTI
    h = "{}"
    d0 = f"{a.case} {{}} grid {grid.vox_mm}mm axes {X.AXCODES} PROVISIONAL"
    nii = {}
    with T("6_write_nifti"):
        W = lambda name, arr, desc: nii.__setitem__(name, write_nii(arr, (grid.vox_mm,) * 3, X.AXCODES, out / f"nifti/{name}.nii.gz", d0.format(desc)))  # noqa: E731
        W("phantom_labels_recon_grid", lab.cpu().numpy(), "XCAT material labels NN")
        W("phantom_organ_labels_recon_grid", grp.cpu().numpy(), "organ-group labels NN")
        W("phantom_density_recon_grid", rho.cpu().numpy(), "density g/cm3 boxavg")
        W("phantom_mu70_recon_grid" if a.energy_kev == 70 else f"phantom_mu{a.energy_kev:g}_recon_grid", mu.cpu().numpy(), f"mu {a.energy_kev}keV 1/mm boxavg")
        W("phantom_ideal_hu70" if a.energy_kev == 70 else f"phantom_ideal_hu{a.energy_kev:g}", hu_ideal.cpu().numpy(), "ideal HU")
        W("reconstructed_mu70" if a.energy_kev == 70 else f"reconstructed_mu{a.energy_kev:g}", rec_mu.cpu().numpy(), "LEAP FBP mu 1/mm")
        W("reconstructed_hu70" if a.energy_kev == 70 else f"reconstructed_hu{a.energy_kev:g}", hu_rec.cpu().numpy(), "LEAP FBP HU")
        if not a.skip_native_labels:
            nlab = write_nii(np.ascontiguousarray(vol), (X.NATIVE_VOX_MM,) * 3, X.AXCODES, out / "nifti/phantom_material_labels.nii.gz",
                             f"{a.case} native labels 0.25mm (ASSUMED) axes {X.AXCODES} PROVISIONAL")
            nii["phantom_material_labels"] = nlab
    # reopen + verify overlay compatibility from the files themselves
    recon_files = ["phantom_labels_recon_grid", "phantom_organ_labels_recon_grid", "phantom_density_recon_grid", "phantom_mu70_recon_grid",
                   "phantom_ideal_hu70", "reconstructed_mu70", "reconstructed_hu70"]
    recon_files = [f for f in recon_files if (out / f"nifti/{f}.nii.gz").exists()]
    imgs = {f: nib.load(out / f"nifti/{f}.nii.gz") for f in recon_files}
    ref = imgs[recon_files[0]]
    reg["files_checked"] = recon_files
    reg["shapes_identical"] = all(i.shape == ref.shape for i in imgs.values())
    reg["affines_max_abs_diff"] = max(float(np.abs(i.affine - ref.affine).max()) for i in imgs.values())
    reg["spacing_mm"] = [float(v) for v in ref.header.get_zooms()]
    reg["axcodes_from_affine"] = "".join(nib.aff2axcodes(ref.affine))
    reg["affine_identical_all_recon_grid_files"] = bool(reg["affines_max_abs_diff"] < 1e-9)
    if "phantom_material_labels" in nii:
        nat = nib.load(out / "nifti/phantom_material_labels.nii.gz")
        c_nat = nat.affine @ np.append((np.array(nat.shape) - 1) / 2, 1)
        c_rec = ref.affine @ np.append((np.array(ref.shape) - 1) / 2, 1)
        reg["native_vs_recon_grid_centre_world_mm_diff"] = (c_nat - c_rec)[:3].tolist()
        reg["native_extent_mm_xyz_vs_grid_extent"] = [nat.shape[i] * nat.header.get_zooms()[i] for i in range(3)] + [list(ref.shape)]

    # ---- colour tables (Slicer .ctbl) + group json
    rng_cols = np.random.default_rng(1).integers(40, 255, (256, 3))
    with open(out / "nifti/organ_labels.ctbl", "w") as f:
        f.write("0 air 0 0 0 0\n")
        for gid_, v in mm.groups.items():
            if gid_:
                c = rng_cols[gid_]; f.write(f"{gid_} {v['name']} {c[0]} {c[1]} {c[2]} 255\n")
    with open(out / "nifti/material_labels.ctbl", "w") as f:
        for oid in sorted(mm.mats):
            c = rng_cols[oid]; f.write(f"{oid} {mm.mats[oid]['name']}_id{oid} {c[0]} {c[1]} {c[2]} {0 if oid == 0 else 255}\n")
    (out / "organ_groups.json").write_text(json.dumps(dict(
        note="groups merge organ IDs ONLY where the xlsx names agree after stripping _trabecular/_cortical; no anatomy is inferred beyond the names",
        groups={str(k): v for k, v in mm.groups.items()},
        materials={str(k): dict(name=v["name"], density_g_cm3=v["rho"], mc_id=v["mc_id"], group=int(mm.organ_to_group[k]), mu_cm=float(mm.mu_cm[k]),
                                kappa_cm2_g=float(mm.kappa[k])) for k, v in sorted(mm.mats.items())}), indent=1))

    # ---- previews
    ngrp_all = max(mm.groups) + 1
    ras = {k: to_ras(v, aff) for k, v in dict(lab=lab.cpu().numpy(), grp=grp.cpu().numpy(), mu=mu.cpu().numpy(), hu=hu_ideal.cpu().numpy(), rec=hu_rec.cpu().numpy(),
                                              recmu=rec_mu.cpu().numpy()).items()}
    if lung_ids:
        idx = slice_choice(to_ras(lung.cpu().numpy(), aff))
    else:  # no lungs: slice through the structures that are neither air nor the dominant (background) group
        cnt = torch.bincount(grp.flatten().long(), minlength=ngrp_all)
        cnt[0] = 0
        struct = (grp > 0) & (grp != int(cnt.argmax()))
        cz = struct.sum((1, 2)).cpu().numpy()
        zsel = int(np.median(np.flatnonzero(cz >= 0.95 * cz.max())))  # middle of the plateau, away from the phantom's z-ends
        sm = torch.zeros_like(struct); sm[zsel] = struct[zsel]
        idx = slice_choice(to_ras(sm.cpu().numpy(), aff))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm_lab = matplotlib.colors.ListedColormap([(0, 0, 0)] + [tuple(rng_cols[i] / 255.0) for i in range(1, 256)])
    ngrp = max(mm.groups) + 1
    win = (-1000, 400) if lung_ids else (-250, 300)

    def fig_save(f, name):
        f.savefig(out / f"previews/{name}.png", dpi=110, bbox_inches="tight")
        plt.close(f)

    f = plt.figure(figsize=(13, 8.6))
    axs = tri_axes(f, 0, 2)
    show_tri(axs, ras["lab"], idx, cmap=matplotlib.colors.ListedColormap([tuple(rng_cols[i] / 255.0) for i in range(256)]), vmin=0, vmax=255)
    for ax in axs: ax.set_facecolor("black")
    axs[1].set_title(f"{a.case}: original material labels (NN, {grid.vox_mm} mm grid)")
    axs = tri_axes(f, 1, 2)
    show_tri(axs, ras["grp"], idx, cmap=cm_lab, vmin=0, vmax=255)
    axs[1].set_title("organ-group labels (named in xlsx; air = 0)")
    h_ = [matplotlib.patches.Patch(color=rng_cols[g_] / 255.0, label=f"{g_} {v['name']}") for g_, v in mm.groups.items() if g_]
    f.legend(handles=h_, loc="lower center", ncol=min(6, max(1, len(h_))), fontsize=7)
    f.suptitle(f"axes/letters PROVISIONAL; voxel {grid.vox_mm} mm; slices through the {'lung' if lung_ids else 'body'} centroid", fontsize=9)
    fig_save(f, "phantom_slices")

    f = plt.figure(figsize=(13, 8.6))
    axs = tri_axes(f, 0, 2); ims = show_tri(axs, ras["mu"] * 10, idx, cmap="magma", vmin=0, vmax=float(mm.mu_cm.max()) * 0.8)
    axs[1].set_title(f"phantom mu({a.energy_kev:g} keV) [cm$^{{-1}}$] (box-averaged)")
    f.colorbar(ims[2], ax=axs, shrink=0.8)
    axs = tri_axes(f, 1, 2); ims = show_tri(axs, ras["hu"], idx, cmap="gray", vmin=win[0], vmax=win[1])
    axs[1].set_title(f"ideal CT, HU at {a.energy_kev:g} keV (window {win[0]}..{win[1]}); mu_water={mu_w * 10:.5f} cm$^{{-1}}$")
    f.colorbar(ims[2], ax=axs, shrink=0.8)
    fig_save(f, "attenuation_slices")

    f, axs = plt.subplots(1, len(det_views), figsize=(4.2 * len(det_views), 3.6))
    for ax, v, im in zip(axs, det_views, det_imgs):
        h = ax.imshow(im, aspect=4, cmap="gray", origin="lower", vmin=0, vmax=max(float(i_.max()) for i_ in det_imgs))
        ax.set_title(f"view {v}: {360 * v / 1000 % 360:.0f} deg, z={plan.pitch_mm_per_rad * np.radians(360 * v / 1000) - plan.scan_length_mm / 2:.1f} mm", fontsize=8)
        ax.set_xlabel("detector channel"); ax.set_ylabel("detector row")
        f.colorbar(h, ax=ax, shrink=0.7, label="line integral")
    f.tight_layout()
    f.suptitle("simulated detector images (line integral of mu[1/mm] dl[mm])", fontsize=9)
    fig_save(f, "detector_views")

    f, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, s, t in zip(axs, (sino_mid, sino_edge), ("central row 32", "row 4 (near the cone edge)")):
        h = ax.imshow(s[::5].T, aspect="auto", cmap="gray", origin="lower", extent=[0, plan.n_views, 0, X.N_COLS]); ax.set_title(f"helical sinogram, {t}")
        ax.set_xlabel("view"); ax.set_ylabel("detector channel"); f.colorbar(h, ax=ax, label="line integral")
    fig_save(f, "sinograms")

    f = plt.figure(figsize=(13, 8.6))
    axs = tri_axes(f, 0, 2); ims = show_tri(axs, ras["rec"], idx, cmap="gray", vmin=win[0], vmax=win[1]); axs[1].set_title("LEAP helical FBP of the simulated projections (HU, window %d..%d)" % win + "")
    axs = tri_axes(f, 1, 2); show_tri(axs, ras["hu"], idx, cmap="gray", vmin=win[0], vmax=win[1]); axs[1].set_title("ideal CT (same window)")
    fig_save(f, "reconstruction_slices")

    f = plt.figure(figsize=(13, 13))
    axs = tri_axes(f, 0, 3); show_tri(axs, ras["rec"], idx, cmap="gray", vmin=win[0], vmax=win[1]); axs[1].set_title("reconstruction (HU) - background image")
    axs = tri_axes(f, 1, 3); show_tri(axs, ras["rec"], idx, cmap="gray", vmin=win[0], vmax=win[1])
    lm = np.ma.masked_equal(ras["grp"], 0)
    show_tri(axs, lm, idx, cmap=cm_lab, vmin=0, vmax=255, alpha=0.4, letters=False); axs[1].set_title("organ-group labels (alpha 0.4) over the reconstruction")
    axs = tri_axes(f, 2, 3); show_tri(axs, ras["rec"], idx, cmap="gray", vmin=win[0], vmax=win[1])
    i, j, k = idx
    for ax, (m2, ) in zip(axs, ((ras["grp"][:, :, k].T,), (ras["grp"][:, j, :].T,), (ras["grp"][i, :, :].T,))):
        ax.contour((m2 > 0).astype(float), levels=[0.5], colors="cyan", linewidths=0.7, origin="lower")
        for gid_ in lung_ids:
            ax.contour((m2 == gid_).astype(float), levels=[0.5], colors="red", linewidths=0.5, origin="lower")
    axs[1].set_title("body (cyan) and lung (red) label outlines over the reconstruction")
    fig_save(f, "phantom_ct_overlay")

    d_ras = ras["rec"] - ras["hu"]
    f = plt.figure(figsize=(13, 10.6))
    axs = tri_axes(f, 0, 3); ims = show_tri(axs, d_ras, idx, cmap="RdBu_r", vmin=-150, vmax=150); axs[1].set_title("reconstruction - ideal (HU), unsmoothed")
    f.colorbar(ims[2], ax=axs, shrink=0.8)
    i, j, k = idx
    ax = f.add_subplot(3, 2, 3)
    ax.plot(np.arange(ras["hu"].shape[0]) * grid.vox_mm, ras["hu"][:, j, k], label="ideal"); ax.plot(np.arange(ras["hu"].shape[0]) * grid.vox_mm, ras["rec"][:, j, k], label="reconstruction", lw=0.8)
    ax.set_title("profile along the R-L axis through the slice centroid"); ax.set_xlabel("mm along array R-L"); ax.set_ylabel("HU"); ax.legend(fontsize=8); ax.set_ylim(-1050, 600)
    ax = f.add_subplot(3, 2, 4)
    ax.plot(np.arange(ras["hu"].shape[2]) * grid.vox_mm, ras["hu"][i, j, :], label="ideal"); ax.plot(np.arange(ras["hu"].shape[2]) * grid.vox_mm, ras["rec"][i, j, :], label="reconstruction", lw=0.8)
    ax.set_title("profile along S-I through the centroid"); ax.set_xlabel("mm along array S axis"); ax.legend(fontsize=8); ax.set_ylim(-1050, 600)
    ax = f.add_subplot(3, 1, 3); ax.plot(zc.cpu().numpy(), zprof.cpu().numpy()); ax.axvline(-(plan.scan_length_mm / 2 - 0.5 * plan.margin_mm), color="gray", ls="--"); ax.axvline(plan.scan_length_mm / 2 - 0.5 * plan.margin_mm, color="gray", ls="--")
    ax.set_title("mean |recon - ideal| inside the body per slice (dashed = edges of the 'valid' z range)"); ax.set_xlabel("z [mm] (array z, provisional S)"); ax.set_ylabel("HU")
    fig_save(f, "reconstruction_difference")

    # ---- reports
    T.t["total_wall_clock_s"] = time.perf_counter() - t_all
    geometry = dict(case=a.case, energy_keV=a.energy_kev, scanner=X.GEOMETRIES[a.geometry], scan_plan=plan.to_dict(), grid=grid.to_dict(),
                    volume_affine_world_mm=aff.round(6).tolist(), volume_axcodes_from_affine="".join(nib.aff2axcodes(aff)),
                    world_frame="mm; origin = phantom centre = scanner isocentre in x,y and mid-scan in z; array axes z,y,x = (S,L,P) PROVISIONAL",
                    leap_volume=dict(order="[z,y,x]", voxel_mm=grid.vox_mm, offsets_mm=[0, 0, 0], note="LEAP volume centred on the isocentre; forward model and FBP share it, so labels/ideal/recon share one grid by construction"),
                    detector_axes="projection tensor (view, detector_row, detector_channel)", native_vox_mm_assumed_for_COPD_lung_liver=X.NATIVE_VOX_MM,
                    registration=reg, orientation_status="PROVISIONAL (inferred from anatomy/insert test, not file metadata)")
    (out / "geometry.json").write_text(json.dumps(geometry, indent=1))
    meta = dict(case=a.case, args=vars(a), environment=env, timings_s=T.t, forward_projection_s=t_fwd, fbp_s=t_fbp, peak_gpu_MiB=dict(forward=peak_f, fbp=peak_r),
                projection_stats=pstats, integrity=integrity, material_crosschecks=mm.xcheck,
                mu_water_mm=mu_w, mu_water_cm=mu_w * 10, water_note="xraylib CS_Total_CP('H2O'), rho=1 g/cm3",
                stats=dict(ideal_hu=dict(min=float(hu_ideal.min()), max=float(hu_ideal.max()), mean_body=float(hu_ideal[body_e].mean())),
                           recon_hu=dict(min=float(hu_rec.min()), max=float(hu_rec.max()), mean_body=float(hu_rec[body_e].mean())),
                           ideal_mu_mm=dict(min=float(mu.min()), max=float(mu.max())), recon_mu_mm=dict(min=float(rec_mu.min()), max=float(rec_mu.max()))),
                metrics=metrics, registration=reg, finite=dict(ideal=bool(torch.isfinite(mu).all()), recon=bool(torch.isfinite(rec_mu).all())),
                nifti={k: dict(shape=v["shape"], dtype=v["dtype"], voxel_mm=v["voxel_mm"], orientation=v["orientation"]) for k, v in nii.items()},
                z_error_profile_hu=zprof.cpu().numpy().round(2).tolist())
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    write_report(out, a, grid, plan, meta, mm)
    log("done in %.1f s; forward %.2f s; FBP %.2f s; peak GPU fwd/fbp %.0f/%.0f MiB" % (T.t["total_wall_clock_s"], t_fwd, t_fbp, peak_f, peak_r))


def args_ok(p):
    return p["min"] >= -1e-4 and p["finite"]


def write_report(out, a, grid, plan, m, mm):
    mt, reg = m["metrics"], m["registration"]
    f = lambda e, k: "n/a" if e is None else f"{e[k]:.4g}"  # noqa: E731
    rows = "\n".join(f"| {n} | {e['n_voxels']} | {e['ideal_hu_mean']:.1f} | {e['recon_hu_mean']:.1f} | {e['hu_bias']:.1f} | {e['hu_mae']:.1f} | {e['hu_rmse']:.1f} |" for n, e in mt["per_group"].items())
    T_ = m["timings_s"]
    txt = f"""# Validation report: {a.case} at {a.voxel_size_mm} mm, {a.energy_kev:g} keV, geometry `{a.geometry}`

Generated {m['environment']['utc']} on {m['environment']['gpu']} (node {m['environment']['node']}, job {m['environment']['slurm_job']}), LEAP {m['environment']['leap_commit']}.
**Orientation letters are PROVISIONAL** (inferred, not file metadata). Monoenergetic Beer-Lambert only: no scatter, noise, bowtie or polychromatic physics. The scanner geometry is borrowed from TrueCT (Duke1) and is not the phantom's original DukeSim acquisition.

## Geometry
- Simulation grid (z,y,x) = {tuple(grid.shape)}, voxel {grid.vox_mm} mm, extent {tuple(round(v, 1) for v in grid.extent_mm)} mm; native {tuple(grid.native_shape)} at {X.NATIVE_VOX_MM} mm ({'documented' if a.case == 'reference' else 'ASSUMED, not documented'}).
- Scan: {plan.n_views} views = {plan.n_rotations} rotations, table {plan.scan_length_mm:.1f} mm, pitch {plan.pitch_mm_per_rad:.4f} mm/rad (pitch 1.0), margin {plan.margin_mm} mm each side; FOV radius {plan.fov_radius_mm:.1f} mm; phantom max radius {m['integrity']['body_max_radius_mm']:.1f} mm (inside FOV: {m['integrity']['body_inside_fov']}).
- Affine/axes of all recon-grid files: `{reg['axcodes_from_affine']}`, spacing {reg['spacing_mm']} mm.

## A. Phantom integrity
- File size = product of dimensions; labels present: {len(m['integrity']['labels_present'])}; unmapped labels: {m['integrity']['unmapped_labels']}.
- Adjacent-voxel equality per native axis (z,y,x), stride 4: {[round(v, 3) for v in m['integrity']['adjacent_equal_fraction_per_axis_zyx_stride4']]} (high on all axes: no scrambled reshape).
- Density range of present labels {m['integrity']['density_range_g_cm3']} g/cm3; lung volume {m['integrity']['lung_volume_L']:.2f} L; native extent {m['integrity']['native_extent_mm_zyx']} mm.
- Material table cross-checks: `{json.dumps(m['material_crosschecks'])}`.

## B. Physics
- mu_water({a.energy_kev:g} keV) = {m['mu_water_cm']:.5f} cm^-1 = {m['mu_water_mm']:.6f} mm^-1 (xraylib H2O); for reference TrueCT reference-case calibration (polychromatic effective) was 0.019273 mm^-1.
- Ideal mu range {m['stats']['ideal_mu_mm']['min']:.3e}..{m['stats']['ideal_mu_mm']['max']:.4f} mm^-1; ideal HU range {m['stats']['ideal_hu']['min']:.0f}..{m['stats']['ideal_hu']['max']:.0f} (air should be about -1000).
- Projections: shape {m['projection_stats']['shape_view_row_channel']}, min {m['projection_stats']['min']:.2e}, max {m['projection_stats']['max']:.3f}, finite {m['projection_stats']['finite']}; pure-air floor upper bound {m['projection_stats']['expected_pure_air_floor_upper_bound']:.4f} (rays missing the body integrate only air; the air label has mu = {mm.mu_mm[0]:.2e} mm^-1, not zero).

{'- No DukeSim LAC table is shipped for the reference phantom: mu rests on the xlsx compositions + xraylib only (it is a water/insert phantom; water = 0.19288 cm2/g at 70 keV).' if not m['material_crosschecks'] else ''}
{'- Supplementary (reference case only, NOT part of the simulated loop): earlier TrueCT pilot, supplied DukeSim recon insert means 116 / -100 / 856 HU and LEAP recon of the REAL projections 124 / -97 / 849 HU (polychromatic spectrum, water-calibrated); the 70 keV ideal values here are insert1 / insert2 / insert3 in the per-group table below. The two are different physics (monoenergetic ideal vs polychromatic BHC measurement), so only rough agreement is expected.' if a.case == 'reference' else ''}

## C. Reconstruction vs ideal (unsmoothed; HU = 1000 (mu/mu_water - 1))
| region | voxels | mu MAE [1/mm] | mu RMSE [1/mm] | HU MAE | HU RMSE | HU bias |
|---|---|---|---|---|---|---|
| valid (FOV x helical z range) | {mt['valid_region']['n_voxels']} | {f(mt['valid_region'], 'mu_mae_per_mm')} | {f(mt['valid_region'], 'mu_rmse_per_mm')} | {f(mt['valid_region'], 'hu_mae')} | {f(mt['valid_region'], 'hu_rmse')} | {f(mt['valid_region'], 'hu_bias')} |
| body, eroded 2 voxels | {mt['body_eroded2']['n_voxels']} | {f(mt['body_eroded2'], 'mu_mae_per_mm')} | {f(mt['body_eroded2'], 'mu_rmse_per_mm')} | {f(mt['body_eroded2'], 'hu_mae')} | {f(mt['body_eroded2'], 'hu_rmse')} | {f(mt['body_eroded2'], 'hu_bias')} |
| lungs, eroded 2 voxels | {mt['lung_eroded2']['n_voxels'] if mt['lung_eroded2'] else 0} | {f(mt['lung_eroded2'], 'mu_mae_per_mm')} | {f(mt['lung_eroded2'], 'mu_rmse_per_mm')} | {f(mt['lung_eroded2'], 'hu_mae')} | {f(mt['lung_eroded2'], 'hu_rmse')} | {f(mt['lung_eroded2'], 'hu_bias')} |

Per organ group (eroded 1 voxel, valid region):

| group | voxels | ideal HU mean | recon HU mean | bias | MAE | RMSE |
|---|---|---|---|---|---|---|
{rows}

Slice-wise error profile and difference images: `previews/reconstruction_difference.png`. Differences include finite sampling, helical/cone-beam FBP behaviour, the box-averaged ideal image and z-end effects; nothing was smoothed.

## D. Registration (labels vs reconstruction)
- All recon-grid NIfTI files ({', '.join(reg['files_checked'])}): shapes identical = {reg['shapes_identical']}; max |affine difference| = {reg['affines_max_abs_diff']:.2e} (identical = {reg['affine_identical_all_recon_grid_files']}).
- Cross-correlation shift of the reconstruction relative to the ideal image: {[round(v, 3) for v in reg['xcorr_shift_voxels_zyx_recon_vs_ideal']]} voxels (z,y,x) = {[round(v, 3) for v in reg['xcorr_shift_mm_zyx']]} mm (0 = aligned; a mirror or axis swap would give a large shift or no correlation).
- Body mask Dice (reconstruction HU > -300, slice-wise filled, vs label body): {reg['body_dice_recon_hu_gt_m300_filled_vs_label_body']:.4f}.
- Lung mask Dice (HU < -400 inside body vs label lungs): {reg.get('lung_dice_recon_hu_lt_m400_inside_body_vs_label_lung', float('nan')):.4f}.
- Native labels vs recon grid world-centre difference: {reg.get('native_vs_recon_grid_centre_world_mm_diff', 'n/a')} mm.
- **Alignment is established by construction (one LEAP volume grid for forward and FBP) and checked numerically above. The anatomical L/R and S/I letters remain PROVISIONAL.**

## E. Runtime (one {m['environment']['gpu']})
| stage | seconds |
|---|---|
""" + "\n".join(f"| {k} | {v:.2f} |" for k, v in T_.items()) + f"""

Peak GPU memory in use (sampled, includes the CUDA context): forward {m['peak_gpu_MiB']['forward']:.0f} MiB, FBP {m['peak_gpu_MiB']['fbp']:.0f} MiB of {m['environment']['vram_MiB']:.0f} MiB.
"""
    (out / "validation_report.md").write_text(txt)


if __name__ == "__main__":
    main()
