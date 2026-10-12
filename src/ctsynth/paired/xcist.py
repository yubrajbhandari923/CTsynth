"""XCIST (gecatsim) adapter: CTsynth fraction phantoms -> XCIST simulation -> arrays + provenance.

gecatsim is an OPTIONAL dependency (BSD-3, sdist-only; use an isolated env, see docs/xcist.md)
imported lazily, so importing this module never needs it. XCIST is a forward-data producer:
material truth is always the input phantom, never anything XCIST returns.

Verified conventions (tests/paired/test_xcist.py):
- raw phantom files are float32 little-endian with x fastest, i.e. a C-order `(iy, ix)` array;
- image row 0 is +y (world y = -(iy-(n-1)/2)*dx), x grows with the column index;
- source at (0, +SID) at view 0, gantry angle grows counter-clockwise; detector arc centred on the
  source, column index grows towards +x at view 0.
Only single-row (2D) scans are supported: multi-row output axis order is unverified.
"""

import contextlib
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

XCIST_FRAME = "xcist-scanner-mm"


def xcist_affine(n: int, dx_mm: float, dz_mm: float) -> tuple[tuple[float, ...], ...]:
    """paired-ct affine [ix, iy, iz, 1] -> [x, y, z, 1] mm, XCIST scanner frame (y flipped)."""
    c = (n - 1) / 2
    return (
        (dx_mm, 0.0, 0.0, -c * dx_mm),
        (0.0, -dx_mm, 0.0, c * dx_mm),
        (0.0, 0.0, dz_mm, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def write_voxel_phantom(
    work: Path,
    stem: str,
    fractions: np.ndarray,
    xnames: list[str],
    dx_mm: float,
    dz_mm: float = 16.0,
) -> Path:
    """Write an XCIST voxelized phantom (json + one raw volume-fraction map per material)."""
    n_mat, nz, ny, nx = fractions.shape
    if nz != 1 or ny != nx or n_mat != len(xnames):
        raise ValueError(f"need (n_mat,1,n,n) fractions matching names, got {fractions.shape}")
    if (fractions < 0).any() or fractions.sum(0).max() > 1 + 1e-6:
        raise ValueError("fractions must be >= 0 and sum to <= 1 (remainder is air)")
    files = []
    for m, name in enumerate(xnames):
        fn = f"{stem}_{name}_{nx}x{ny}x1.raw"
        np.ascontiguousarray(fractions[m, 0], dtype="<f4").tofile(work / fn)
        files.append(fn)
    k = n_mat
    vp = {
        "n_materials": k,
        "mat_name": list(xnames),
        "volumefractionmap_filename": files,
        "volumefractionmap_datatype": ["float"] * k,
        "cols": [nx] * k,
        "rows": [ny] * k,
        "slices": [1] * k,
        "x_size": [dx_mm] * k,
        "y_size": [dx_mm] * k,
        "z_size": [dz_mm] * k,
        "x_offset": [(nx + 1) / 2] * k,
        "y_offset": [(ny + 1) / 2] * k,
        "z_offset": [1.0] * k,
    }
    (work / f"{stem}.json").write_text(json.dumps(vp, indent=1))
    (work / f"Phantom_{stem}.cfg").write_text(
        'phantom.callback = "Phantom_Voxelized"\n'
        'phantom.projectorCallback = "C_Projector_Voxelized"\n'
        f'phantom.filename = "{stem}.json"\nphantom.centerOffset = [0.0, 0.0, 0.0]\n'
        "phantom.scale = 1.0\n"
    )
    return work / f"Phantom_{stem}"


@dataclass(frozen=True)
class XcistAcq:
    kvp: int | None = 120  # spectrum file tungsten_tar7.0_{kvp}_filt.dat; ignored if mono_keV set
    mono_keV: float | None = None
    mA: float = 800.0
    n_view: int = 72
    n_col: int = 900
    col_size_mm: float = 1.0
    sid_mm: float = 540.0
    sdd_mm: float = 950.0
    col_offset: float = 0.25  # detector columns, XCIST default
    quantum_noise: bool = False
    electronic_noise: bool = False
    subsample: bool = False  # False: 1 sample/axis (isolates geometry); True: XCIST default 2
    shaped_beam: bool = False  # True keeps XCIST's default bowtie + 3 mm Al flat filter
    seed: int = 0


def fan_rays(a: XcistAcq) -> tuple[np.ndarray, np.ndarray]:
    """Central rays implied by the verified convention: src (n_view,2), det (n_view,n_col,2) mm."""
    d_alpha = 2 * np.arctan(a.col_size_mm / 2 / a.sdd_mm)
    u_off = np.arctan(a.col_offset * a.col_size_mm / a.sdd_mm)
    al = (np.arange(a.n_col) - (a.n_col - 1) / 2) * d_alpha + u_off
    d0 = np.stack([a.sdd_mm * np.sin(al), a.sid_mm - a.sdd_mm * np.cos(al)], axis=1)
    b = 2 * np.pi * np.arange(a.n_view) / a.n_view
    rot = np.stack(
        [np.stack([np.cos(b), -np.sin(b)], -1), np.stack([np.sin(b), np.cos(b)], -1)], -2
    )
    src = np.einsum("vij,j->vi", rot, np.array([0.0, a.sid_mm]))
    det = np.einsum("vij,cj->vci", rot, d0)
    return src, det


@contextlib.contextmanager
def _chdir(p: Path) -> Iterator[None]:
    old = os.getcwd()
    os.chdir(p)
    try:
        yield
    finally:
        os.chdir(old)


def run_xcist(
    work: Path,
    fractions: np.ndarray,
    xnames: list[str],
    dx_mm: float,
    acq: XcistAcq,
    *,
    recon: bool = False,
) -> dict[str, Any]:
    """Simulate (and optionally FDK-reconstruct) one phantom. Returns arrays + provenance dict.

    Raises FileNotFoundError naming any expected XCIST output that was not produced.
    """
    import gecatsim as xc  # optional dependency

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    pkg = Path(xc.__file__).parent
    shutil.copytree(pkg / "examples" / "cfg", work / "cfg", dirs_exist_ok=True)
    write_voxel_phantom(work, "ph", fractions, xnames, dx_mm)
    with _chdir(work):
        xc.pyfiles.CommonTools.my_path.add_search_path(str(work))
        ct = xc.CatSim(
            "./Phantom_ph", "./cfg/Scanner_Sample_generic", "./cfg/Protocol_Sample_axial"
        )
        ct.resultsName = "out"
        pr, sc, ph = ct.protocol, ct.scanner, ct.physics
        pr.viewsPerRotation = pr.viewCount = acq.n_view
        pr.stopViewId = acq.n_view - 1
        pr.mA = acq.mA
        sc.detectorRowsPerMod = sc.detectorRowCount = 1
        sc.detectorColCount, sc.detectorColSize = acq.n_col, acq.col_size_mm
        sc.sid, sc.sdd, sc.detectorColOffset = acq.sid_mm, acq.sdd_mm, acq.col_offset
        if acq.mono_keV is not None:
            ph.monochromatic = acq.mono_keV
        else:
            pr.spectrumFilename = f"tungsten_tar7.0_{acq.kvp}_filt.dat"
        if not acq.shaped_beam:
            pr.bowtie, pr.flatFilter = [], []
        ph.enableQuantumNoise, ph.enableElectronicNoise = (
            int(acq.quantum_noise),
            int(acq.electronic_noise),
        )
        n_sub = 2 if acq.subsample else 1
        for k in (
            "colSampleCount",
            "rowSampleCount",
            "srcXSampleCount",
            "srcYSampleCount",
            "viewSampleCount",
        ):
            setattr(ph, k, n_sub)
        for k in ("col_crosstalk", "row_crosstalk", "col_crosstalk_opt", "row_crosstalk_opt"):
            setattr(ph, k, 0)
        np.random.seed(acq.seed)  # noqa: NPY002  # electronic noise uses the global NumPy RNG
        if hasattr(ct.cfg, "clib"):
            # quantum-noise RNG lives in the C library; it exits the process on a seed of 0
            ct.cfg.clib.setall(acq.seed + 1, acq.seed + 2)
        ct.run_all()
        names = {
            "air": (1, acq.n_col),
            "offset": (1, acq.n_col),
            "scan": (acq.n_view, acq.n_col),
            "prep": (acq.n_view, acq.n_col),
        }
        out: dict[str, Any] = {}
        for ext, shape in names.items():
            f = Path(f"out.{ext}")
            if not f.exists():
                raise FileNotFoundError(f"XCIST did not produce {f} in {work}")
            out[ext] = np.fromfile(f, dtype="<f4").reshape(shape).astype(np.float64)
        spec = ct.cfg.spec
        net = np.atleast_2d(np.asarray(spec.netIvec, dtype=float))
        w = np.atleast_2d(np.asarray(ct.cfg.sim.Wvec, dtype=float))
        # gecatsim keeps these per detector cell; with an unshaped beam and a curved detector
        # focused on the source every cell must be identical, else a single row would be a lie
        if not (np.allclose(net, net[0], rtol=1e-6) and np.allclose(w, w[0], rtol=1e-6)):
            raise ValueError("spectrum/detector weights vary across cells: use shaped_beam=False")
        from gecatsim.pyfiles.GetMu import GetMu
        from gecatsim.pyfiles.ReadMaterialFile import ReadMaterialFile

        ev = np.atleast_1d(spec.Evec).astype(np.single)
        mats = {m: [float(v) for v in GetMu(m, ev)] for m in xnames}  # linear mu, 1/cm
        dens = {
            m: float(ReadMaterialFile(xc.pyfiles.CommonTools.my_path.find("material", m, ""))[1])
            for m in xnames
        }
        out["spectrum"] = {
            "energies_keV": ev.astype(float).tolist(),
            "net_fluence": net[0].tolist(),
            "detector_weights": w[0].tolist(),
            "mu_per_cm": mats,
            "density_g_cm3": dens,
        }
        if recon:
            import gecatsim.reconstruction.pyfiles.recon as rc

            ct.recon.fov, ct.recon.imageSize, ct.recon.sliceCount = (
                fractions.shape[-1] * dx_mm,
                fractions.shape[-1],
                1,
            )
            ct.recon.sliceThickness = 1.0
            ct.do_Recon = 1
            rc.recon(ct)
            n = fractions.shape[-1]
            out["recon"] = np.fromfile(f"out_{n}x{n}x1.raw", dtype="<f4").reshape(n, n)
    out["provenance"] = {
        "gecatsim": metadata.version("gecatsim"),
        "numpy": np.__version__,
        "acq": acq.__dict__,
        "xcist_names": xnames,
        "dx_mm": dx_mm,
        "cfg_files": sorted(p.name for p in (work / "cfg").glob("*.cfg")),
    }
    return out


def to_paired_record(
    out: dict[str, Any],
    fractions: np.ndarray,
    names: tuple[str, ...],
    ref_density: np.ndarray,
    acq: XcistAcq,
    *,
    phantom_id: str,
    acquisition_id: str,
    dest: Path,
    attenuation_source: str,
    extra_provenance: dict[str, Any] | None = None,
) -> Path:
    """Save one XCIST run as a paired-ct record; truth is the INPUT `fractions`."""
    from ctsynth.paired.schema import (
        Grid,
        MaterialSpec,
        Measurement,
        PairedRecord,
        npy_sha256,
        save_record,
    )

    prov = out["provenance"]
    n = fractions.shape[-1]
    mono = f"monochromatic-{acq.mono_keV}keV" if acq.mono_keV else "polychromatic-EI"
    arrays: dict[str, tuple[np.ndarray, str, tuple[str, ...], str, str]] = {
        "prep": (out["prep"], "line_integral_prep", ("view", "col"), "1 (-ln ratio, clip 9)", mono),
        "air": (out["air"], "signal_air", ("row", "col"), "XCIST-DAS-units(unverified)", mono),
        "scan": (out["scan"], "signal_scan", ("view", "col"), "XCIST-DAS-units(unverified)", mono),
    }
    if "recon" in out:
        arrays["recon"] = (out["recon"], "recon_hu", ("y", "x"), "HU(nominal mu 0.02/mm)", mono)
    meas = tuple(
        Measurement(k, kind, f"{k}.npy", v.shape, str(v.dtype), ax, units, es, npy_sha256(v))  # type: ignore[arg-type]
        for k, (v, kind, ax, units, es) in arrays.items()
    )
    unknown = [] if "recon" in out else ["recon"]  # scatter is OFF here: a known value
    rec = PairedRecord(
        record_id=f"{phantom_id}:{acquisition_id}",
        phantom_id=phantom_id,
        acquisition_id=acquisition_id,
        producer={"name": "XCIST/gecatsim", "version": prov["gecatsim"]},
        grid=Grid((1, n, n), xcist_affine(n, prov["dx_mm"], 16.0), XCIST_FRAME),
        materials=tuple(
            MaterialSpec(nm, "volume_fraction", float(r), attenuation_source)
            for nm, r in zip(names, ref_density, strict=True)
        ),
        truth_file="truth.npy",
        truth_sha256=npy_sha256(fractions),
        sum_constraint="le1",
        acquisition={
            "geometry": {
                "type": "fan-third-gen-curved-1row",
                "sid_mm": acq.sid_mm,
                "sdd_mm": acq.sdd_mm,
                "n_view": acq.n_view,
                "n_col": acq.n_col,
                "col_size_mm": acq.col_size_mm,
                "col_offset": acq.col_offset,
            },
            "spectrum": out["spectrum"]
            | {"kvp": acq.kvp, "mono_keV": acq.mono_keV, "shaped_beam": acq.shaped_beam},
            "detector": {"type": "energy-integrating", "model": "XCIST Detection_EI (Lumex 3mm)"},
            "scatter": {"model": "none"},
            "noise": {
                "quantum": acq.quantum_noise,
                "electronic": acq.electronic_noise,
                "seed": acq.seed,
            },
            "recon": ({"algorithm": "FDK", "unit": "HU"} if "recon" in out else None),
        },
        unknown=tuple(unknown),
        measurements=meas,
        provenance=prov | (extra_provenance or {}),
    )
    return save_record(rec, fractions, {k: v[0] for k, v in arrays.items()}, dest)
