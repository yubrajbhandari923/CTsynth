"""XCAT material phantom -> attenuation -> LEAP projections -> LEAP helical FBP: reusable pieces (no CLI here).

Conventions (see geometry.json written by the pipeline):
 * Phantom .bin files are C-order uint8 [z, y, x]; array axes are PROVISIONALLY (S, L, P) i.e. each index increases toward
   Superior / patient-Left / Posterior (reference: from the insert-HU test; COPD: from anatomical markers; see README). The same
   array layout is used on the simulation grid, so labels, ideal CT and reconstruction share ONE voxel grid and ONE affine.
 * World frame: millimetres, origin at the phantom's physical centre, which is placed on the scanner isocentre / mid-scan.
 * Attenuation: mu(r, E) = rho(r) * sum_i w_i (mu/rho)_i(E)  [cm^-1], elemental mass fractions w_i from the case's xlsx,
   (mu/rho)_i = xraylib CS_Total (cm^2/g, incl. coherent). LEAP wants mm^-1: mu_mm = mu_cm / 10.
 * Projection tensor axes: (view, detector_row, detector_channel), float32, dimensionless line integrals of mu[mm^-1] dl[mm].
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openpyxl
import torch

from common import INFILES, PHANTOMS, ROOT, array_shape

HERE = Path(__file__).resolve().parent
AXCODES = "SLP"  # array axes (z, y, x): PROVISIONAL, see module docstring
NATIVE_VOX_MM = 0.25  # documented in the reference filename only; ASSUMED for COPD / lung / liver (README)
SOD, SDD, PIX_MM, N_ROWS, N_COLS = 575.0, 1050.0, 1.0, 64, 900
GEOMETRIES = {  # initial scanner configuration: the TrueCT Duke1 helical geometry, ideal (no empirical real-data offsets)
    "truect_helical": dict(sod=SOD, sdd=SDD, pixel_mm=PIX_MM, rows=N_ROWS, cols=N_COLS, views_per_rotation=1000,
                           table_mm_per_rotation=35.05, curved_detector=True, centre_row=(N_ROWS - 1) / 2, centre_col=(N_COLS - 1) / 2,
                           note="Duke1 per PDF Table 1 / DICOM-CT-PD tags: pitch 1.0 (35.05 mm per rotation); borrowed, NOT the phantom's original DukeSim acquisition; "
                                "the empirical real-data offsets of recon_final.py (-1 channel, volume offset) are NOT applied"),
}
LAC_BINS = {"copd_22728": ROOT / "COPD/material_lac/22728/lin_att_120kV_121materials.bin",
            "copd_21535": ROOT / "COPD/material_lac/21535/lin_att_120kV_121materials.bin",
            "lung_18460": ROOT / "LungLesions/material_lac/18460/lin_att_120kV_121materials.bin",
            "liver_279": ROOT / "LiverLesions/material_lac/pt279/lin_att_120kV_121materials.bin"}


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        outer = self

        class _C:
            def __enter__(s):
                s.t0 = time.perf_counter()

            def __exit__(s, *a):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                outer.t[name] = outer.t.get(name, 0.0) + time.perf_counter() - s.t0
        return _C()


# ----------------------------------------------------------------------------- materials
def load_materials(xlsx):
    """Organ ID -> dict(name, rho g/cm3, mc_id, w{Z: mass fraction}) from a TrueCT material xlsx."""
    rows = list(openpyxl.load_workbook(xlsx, data_only=True).active.iter_rows(values_only=True))
    hdr = rows[0]
    ci = {h: i for i, h in enumerate(hdr)}
    zs = [h for h in hdr if isinstance(h, int)]
    out = {}
    for r in rows[1:]:
        if r[0] is None:
            continue
        out[int(r[ci["Organ ID"]])] = dict(name=str(r[0]), rho=float(r[ci["Density"]]), mc_id=int(r[ci["MC ID"]]),
                                           w={z: float(r[ci[z]] or 0.0) for z in zs})
    return out


class ElementTable:
    def __init__(self, path=HERE / "data/element_mass_atten_xraylib.json"):
        d = json.loads(Path(path).read_text())
        self.E = np.array(d["energies_keV"])
        self.k = {int(z): np.array(v) for z, v in d["elements"].items()}
        self.water = np.array(d["water_H2O_CS_Total_CP"])
        self.meta = {k: d[k] for k in ("quantity", "xraylib_version", "caveats")}

    def _interp(self, y, E):
        if not (self.E[0] <= E <= self.E[-1]):
            raise ValueError(f"energy {E} keV outside the table {self.E[0]}-{self.E[-1]}")
        return float(np.exp(np.interp(math.log(E), np.log(self.E), np.log(y))))

    def kappa(self, z, E):
        return self._interp(self.k[z], E)

    def kappa_water(self, E):
        return self._interp(self.water, E)


def organ_group(name):
    n = name.lower()
    for suf in ("_trabecular", "_cortical"):
        n = n.removesuffix(suf)
    return n


def group_category(name):
    if name == "air":
        return "air"
    if name.startswith("insert"):
        return "phantom_insert"
    if "lesion" in name:
        return "lesion"
    if name == "emphysema":
        return "pathology"
    return "tissue_named_in_xlsx"


@dataclass
class MaterialModel:
    energy_keV: float
    mats: dict
    rho: np.ndarray            # (256,) g/cm3, 0 for unmapped
    kappa: np.ndarray          # (256,) cm2/g at energy_keV
    mu_cm: np.ndarray          # (256,) cm^-1
    mu_water_mm: float         # mm^-1 (xraylib H2O, rho = 1 g/cm3)
    groups: dict = field(default_factory=dict)       # group id -> dict(name, organ_ids, ...)
    organ_to_group: np.ndarray = None                 # (256,) uint8, 0 = air/background
    xcheck: dict = field(default_factory=dict)

    @property
    def mu_mm(self):
        return (self.mu_cm / 10.0).astype(np.float32)


def build_material_model(case, energy_keV, elem=None):
    elem = elem or ElementTable()
    mats = load_materials(PHANTOMS[case][2])
    rho, kap, mu = np.zeros(256), np.zeros(256), np.zeros(256)
    for oid, m in mats.items():
        sw = sum(m["w"].values())  # some rows sum to 0.96-0.999: normalised, as DukeSim's own table is (checked below)
        kap[oid] = sum(w * elem.kappa(z, energy_keV) for z, w in m["w"].items()) / sw
        rho[oid] = m["rho"]
        mu[oid] = kap[oid] * rho[oid]
    kw = elem.kappa_water(energy_keV)
    names = sorted({organ_group(m["name"]) for m in mats.values()}, key=lambda n: min(o for o, m in mats.items() if organ_group(m["name"]) == n))
    gid = {"air": 0}
    for n in names:
        if n != "air":
            gid[n] = len(gid)
    o2g = np.zeros(256, np.uint8)
    groups = {0: dict(name="air", organ_ids=[], member_names=["air"], category="air")}
    for oid, m in sorted(mats.items()):
        g = gid[organ_group(m["name"])]
        o2g[oid] = g
        e = groups.setdefault(g, dict(name=organ_group(m["name"]), organ_ids=[], member_names=[], category=group_category(organ_group(m["name"]))))
        e["organ_ids"].append(oid)
        if m["name"] not in e["member_names"]:
            e["member_names"].append(m["name"])
    mm = MaterialModel(energy_keV, mats, rho, kap, mu, kw / 10.0, groups, o2g)
    mm.xcheck = crosscheck_dukesim_lac(case, mm)
    return mm


def crosscheck_dukesim_lac(case, mm):
    """Compare our composition-based mu(E) per organ with DukeSim's own lin_att_120kV_121materials.bin (rows = organ ID, columns =
    1..120 keV integer, cm^-1); and the xlsx vs conversion_table .in. Independent source: agreement is evidence, not a tautology."""
    out = {}
    E = mm.energy_keV
    p = LAC_BINS.get(case)
    if p is not None and Path(p).exists() and abs(E - round(E)) < 1e-9 and 1 <= round(E) <= 120:
        tab = np.fromfile(p, dtype=np.float32).reshape(121, 120)
        ids = sorted(mm.mats)
        ref = tab[ids, int(round(E)) - 1].astype(np.float64)
        ours = mm.mu_cm[ids]
        rel = np.abs(ours - ref) / np.maximum(ref, 1e-12)
        out["dukesim_lac_bin"] = dict(file=str(p), energy_keV=E, n_organs=len(ids), max_rel_diff=float(rel.max()),
                                      worst_organ_id=int(ids[int(rel.argmax())]), median_rel_diff=float(np.median(rel)))
    inf = INFILES.get(case)
    if inf is not None and Path(inf).exists():
        cross = {}
        for ln in Path(inf).read_text().splitlines():
            m = re.match(r"^(\d+)\s+(\S+)\s+(\d+)\s+([0-9.eE+-]+)", ln)
            if m:
                cross[int(m[1])] = (m[2], float(m[4]), int(m[3]))
        bad = [k for k, v in mm.mats.items() if k not in cross or cross[k][0] != v["name"] or abs(cross[k][1] - v["rho"]) > 1e-9 or cross[k][2] != v["mc_id"]]
        out["xlsx_vs_conversion_table_in"] = dict(file=str(inf), n_mismatch=len(bad), mismatched_ids=bad)
    return out


# ----------------------------------------------------------------------------- phantom + grids
def open_phantom(case):
    path, dims, xlsx, _ = PHANTOMS[case]
    shape = array_shape(dims)
    size = Path(path).stat().st_size
    if size != int(np.prod(shape)):
        raise ValueError(f"{path}: size {size} != prod{shape} for uint8")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=shape), shape


@dataclass
class Grid:
    shape: tuple          # (nz, ny, nx)
    vox_mm: float
    native_shape: tuple
    native_vox_mm: float
    axcodes: str = AXCODES

    @property
    def extent_mm(self):
        return tuple(n * self.vox_mm for n in self.shape)

    def to_dict(self):
        return dict(shape_zyx=list(self.shape), vox_mm=self.vox_mm, extent_mm_zyx=list(self.extent_mm), native_shape_zyx=list(self.native_shape),
                    native_vox_mm=self.native_vox_mm, array_axis_directions_zyx=list(self.axcodes))


def make_grid(native_shape, native_vox, vox, pad_vox=2):
    n = tuple(int(math.ceil(s * native_vox / vox - 1e-9)) + pad_vox for s in native_shape)
    return Grid(n, float(vox), tuple(native_shape), float(native_vox))


def affine_for(shape, vox_mm, axcodes=AXCODES):
    from niiutil import affine_from_axcodes
    return affine_from_axcodes(shape, (vox_mm,) * 3, axcodes)


def box_weights(n_out, vox_out, n_in, vox_in, device):
    """(n_out, n_in) matrix: overlap length of output voxel j with input voxel i divided by vox_out. Grids share the same centre.
    Row sums = fraction of the output voxel covered by the phantom (1 inside, <1 on the padded border)."""
    off = (n_out * vox_out - n_in * vox_in) / 2.0
    lo_o = torch.arange(n_out, dtype=torch.float64) * vox_out - off
    lo_i = torch.arange(n_in, dtype=torch.float64) * vox_in
    ov = (torch.minimum(lo_o[:, None] + vox_out, lo_i[None] + vox_in) - torch.maximum(lo_o[:, None], lo_i[None])).clamp(min=0) / vox_out
    return ov.float().to(device)


def nn_index(n_out, vox_out, n_in, vox_in):
    """Native index under the CENTRE of each output voxel (nearest neighbour) and a validity flag (inside the native extent)."""
    off = (n_out * vox_out - n_in * vox_in) / 2.0
    c = (np.arange(n_out) + 0.5) * vox_out - off
    i = np.floor(c / vox_in).astype(np.int64)
    ok = (i >= 0) & (i < n_in)
    return np.clip(i, 0, n_in - 1), ok


def resample_phantom(vol, grid: Grid, mm: MaterialModel, device, chunk=48):
    """Native labels -> (labels NN, organ-group labels NN, density g/cm3 box-mean, mu_mm box-mean, native label histogram).
    mu and density are box-averaged (exact overlap weights, any size ratio); labels are nearest-neighbour; the padded border is air."""
    nz, ny, nx = grid.shape
    nzn, nyn, nxn = grid.native_shape
    Wz, Wy, Wx = (box_weights(n, grid.vox_mm, m, grid.native_vox_mm, device) for n, m in zip(grid.shape, grid.native_shape))
    iz, okz = nn_index(nz, grid.vox_mm, nzn, grid.native_vox_mm)
    iy, oky = nn_index(ny, grid.vox_mm, nyn, grid.native_vox_mm)
    ix, okx = nn_index(nx, grid.vox_mm, nxn, grid.native_vox_mm)
    lut_mu = torch.from_numpy(mm.mu_mm).to(device)
    lut_rho = torch.from_numpy(mm.rho.astype(np.float32)).to(device)
    lut_grp = torch.from_numpy(mm.organ_to_group.astype(np.uint8)).to(device)
    mu = torch.zeros(grid.shape, device=device)
    rho = torch.zeros(grid.shape, device=device)
    lab = torch.zeros(grid.shape, dtype=torch.uint8, device=device)
    hist = torch.zeros(256, dtype=torch.int64, device=device)
    iy_t, ix_t = torch.from_numpy(iy).to(device), torch.from_numpy(ix).to(device)
    mask_yx = torch.from_numpy(np.outer(oky, okx)).to(device)
    for z0 in range(0, nzn, chunk):
        z1 = min(z0 + chunk, nzn)
        L = torch.from_numpy(np.ascontiguousarray(vol[z0:z1])).to(device)
        hist += torch.bincount(L.flatten().long(), minlength=256)
        Li = L.long()
        for src, lut, dst in ((Li, lut_mu, mu), (Li, lut_rho, rho)):
            a = lut[src]
            a = torch.einsum("oy,zyx->zox", Wy, a)
            a = torch.einsum("px,zox->zop", Wx, a)
            dst += torch.einsum("kz,zop->kop", Wz[:, z0:z1], a)
        ks = np.flatnonzero((iz >= z0) & (iz < z1) & okz)
        if ks.size:
            sl = L[torch.from_numpy(iz[ks] - z0).to(device)]            # (k, ny_nat, nx_nat)
            sl = sl[:, iy_t][:, :, ix_t]
            sl = torch.where(mask_yx[None], sl, torch.zeros_like(sl))
            lab[torch.from_numpy(ks).to(device)] = sl
        del L, Li
    cov = Wz.sum(1)[:, None, None] * Wy.sum(1)[None, :, None] * Wx.sum(1)[None, None, :]   # fraction of each voxel inside the phantom
    mu += (1 - cov) * float(mm.mu_mm[0])      # label 0 = air
    rho += (1 - cov) * float(mm.rho[0])
    return lab, lut_grp[lab.long()], rho, mu, hist.cpu().numpy()


def mu_to_hu(mu_mm, mu_water_mm):
    return 1000.0 * (mu_mm / mu_water_mm - 1.0)


# ----------------------------------------------------------------------------- LEAP
@dataclass
class ScanPlan:
    geometry: str
    n_rotations: int
    n_views: int
    scan_length_mm: float
    margin_mm: float
    pitch_mm_per_rad: float
    fov_radius_mm: float

    def to_dict(self):
        return self.__dict__.copy()


def plan_scan(grid: Grid, geometry="truect_helical", margin_mm=40.0):
    g = GEOMETRIES[geometry]
    ext_z = grid.shape[0] * grid.vox_mm
    n_rot = int(math.ceil((ext_z + 2 * margin_mm) / g["table_mm_per_rotation"]))
    beta = 0.5 * g["cols"] * g["pixel_mm"] / g["sdd"]
    return ScanPlan(geometry, n_rot, n_rot * g["views_per_rotation"], n_rot * g["table_mm_per_rotation"], margin_mm,
                    g["table_mm_per_rotation"] / (2 * math.pi), g["sod"] * math.sin(beta))


def make_leap(grid: Grid, plan: ScanPlan, device=0, L=None):
    """LEAP model for the ideal TrueCT helical geometry. Volume array [z,y,x], centred on the isocentre / mid-scan."""
    import sys
    sys.path.insert(0, "/home/yb107/ctsynth_truect/leap/LEAP/src")
    from leapctype import tomographicModels
    g = GEOMETRIES[plan.geometry]
    L = L if L is not None else tomographicModels()  # pass leaptorch.Projector().leapct to share the geometry with autograd
    phis = (360.0 * np.arange(plan.n_views) / g["views_per_rotation"]).astype(np.float32)
    L.set_conebeam(plan.n_views, g["rows"], g["cols"], g["pixel_mm"], g["pixel_mm"], g["centre_row"], g["centre_col"], phis, g["sod"], g["sdd"])
    if g["curved_detector"]:
        L.set_curvedDetector()
    L.set_helicalPitch(plan.pitch_mm_per_rad)
    nz, ny, nx = grid.shape
    L.set_volume(nx, ny, nz, grid.vox_mm, grid.vox_mm, 0.0, 0.0, 0.0)
    L.set_gpu(device)
    return L


def forward_project(L, mu_mm: torch.Tensor, plan: ScanPlan):
    g = torch.zeros((plan.n_views, N_ROWS, N_COLS), dtype=torch.float32, device=mu_mm.device)
    L.project(g, mu_mm.contiguous())
    return g


def fbp(L, g: torch.Tensor, grid: Grid, inplace=False):
    f = torch.zeros(grid.shape, dtype=torch.float32, device=g.device)
    L.FBP(g, f, inplace=inplace)
    return f


def write_projection_zarr(g: torch.Tensor, plan: ScanPlan, path: Path, attrs: dict, view_chunk=250):
    import sys
    sys.path.insert(0, str(Path.home() / "ctsynth_truect/pylib"))
    import numcodecs
    import zarr
    path = Path(path)
    if path.exists():
        import shutil
        shutil.rmtree(path)
    z = zarr.open(str(path), mode="w", shape=tuple(g.shape), chunks=(view_chunk, g.shape[1], g.shape[2]), dtype="f4",
                  compressor=numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE))
    for v0 in range(0, g.shape[0], 1000):
        z[v0:v0 + 1000] = g[v0:v0 + 1000].cpu().numpy()
    g_attrs = dict(axes=["view", "detector_row", "detector_channel"], units="dimensionless line integral of mu[1/mm] over path[mm]",
                   lossless=True, dtype="float32", **attrs)
    z.attrs.update(json.loads(json.dumps(g_attrs, default=str)))
    return z


def projection_preview(g: torch.Tensor, kv=5, kc=3):
    """Downsampled (views/kv, rows, channels/kc) tensor for exploration. NOT an anatomical volume."""
    x = g[None, None]
    return torch.nn.functional.avg_pool3d(x, (kv, 1, kc), (kv, 1, kc))[0, 0].cpu().numpy()
