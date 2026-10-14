# AAPM DL-Spectral-CT adapter spec (T-010, frozen with tests/aapm)

Code: `src/ctsynth/aapm/{schema,synthetic,splits,loader,metrics}.py`. Tests: `tests/aapm/` (run without real case data).
Source facts: `docs/literature/aapm/schema-audit.md`; tables `data/aapm_kit_sample/`. **Synthetic data are not AAPM
data and must never be reported as AAPM results.**

## 1. Frozen conventions
| Item | Convention |
|---|---|
| Material order (everywhere) | `adipose, fibroglandular, calcification` (`schema.MATERIALS`); never permuted silently |
| Phantoms on disk | one `Phantom_{Adipose,Fibroglandular,Calcification}.npy`, `(N,512,512)` float32, `[case, ix, iy]` |
| Phantoms in memory | `(N,3,n,n)` float32, material axis 1 in canonical order |
| Images | `{low,high}kVpImages.npy` `(N,512,512)` float32, `image[ix,iy]` (axis0 = x), NOT row/col |
| Transmission | `{low,high}kVpTransmission.npy` `(N,256,1024)` float32, `[case, view_within_kvp, bin]`, in (0,1], I0 = 1 |
| kVp parity | even views (0,2,..) = **80 kVp** (`high`), odd views = **50 kVp** (`low`); 256 each |
| Units | maps: dimensionless volume fraction (INFERRED, not stated by the kit); mu: cm^-1; lengths cm |
| Partition | sum == 0 air (35.1%), sum == 1 tissue (59.6%), strictly between = boundary band (5.2%) (F-013); tol 1e-3; sum <= 1.001 |
Violations raise `SchemaError` with `.convention` in {shape, dtype, range, finite, material_order, kvp_parity, partition, table}.
Content guards: calcification support (>1e-3) must be < 5% of pixels (catches calcification swapped with a bulk
channel); mean T(50 kVp) < mean T(80 kVp) (catches swapped kVp labels). **Adipose <-> fibroglandular cannot be
detected from pixels alone** (they are 1 deg apart in attenuation space, F-012); only names (`material_names=`) or
file names (loader) catch it.

## 2. Tables
`model_data_{50,80}kVp.npy` `(5,E)`: row0 keV, row1 s (`sum 0.5*s = 1`, tol 1e-4: file is float32), rows 2-4 mu
(canonical order). `load_model_table` -> `ModelTable`.

## 3. Geometry (`GeometrySpec`)
Circular fan beam, flat detector, R = 50 cm, SDD = 100 cm, FOV 18 cm square, pixel = fov/n_pix (0.03515625 cm).
View k: angle `s_k = 2*pi*k/n_views` CCW; source `R(cos s, sin s)`; detector centre `(R-SDD)(cos s, sin s)`; detector
axis `e_u = (-sin s, cos s)`. Detector length `L = 2*tan(asin((fov/2)/R))*SDD` (36.598 cm); bin centre
`u_j = -L/2 + (j+0.5)L/n_det`. Pixel `(ix,iy)` covers `x in [-fov/2 + ix*dx, +dx]`, same for y.
Reduced synthetic geometries keep fov/R/SDD and shrink `n_pix, n_views, n_det`.

## 4. Forward model (synthetic and reference)
**Official projector** (`starting_kit/model/projection.py`, ported in `ctsynth.aapm.fanbeam`): per ray, step through the
image layer by layer along the dominant axis; each layer adds `travPixlen*pixel` (one pixel) or splits `travPixlen`
linearly between the two straddled pixels (`travPixlen = d*sqrt(1+slope^2)`). This is NOT exact chord length; the
exact-chord reference (`synthetic.line_integrals`, tests) differs by a discretisation term, quantified in EXP-017.
Kit view order: 512 views, `s = 2*pi*k/512`; kVp-split by index parity (even=80, odd=50).
Exact-chord reference model: `P_m = sum_pixels frac_m * (exact ray-in-pixel chord length, cm)`; `T = sum_E 0.5*s_kVp(E)*exp(-sum_m mu_m(E)*P_m)`;
noiseless. Synthetic images = `sum_m frac_m * sum_E 0.5 s_k mu_m` (spectrum-weighted mu; NOT FBP).

## 5. Splits
Per-case, by index only. `u = ctsynth.seeding.rng(seed, i).random()`; train if `u < f_tr`, val if `u < f_tr+f_va`,
else test (defaults 0.8/0.1/0.1). **Stability:** case i's split is a function of (seed, i, fractions) only, so it is
identical for any N > i (prefix-stable) and across runs; sizes are binomial, not exact. `split_indices(..., "test")`
raises `TestSplitGuardError` unless `purpose="final_evaluation"`. The guard is a speed bump, not security; tuning on
test remains a protocol violation (CLAUDE.md).

## 6. Loader
Directory of the 7 uncompressed `.npy` files (decompress `.gz` first). Each opened with
`np.load(mmap_mode="r")`; `CaseStore[i]` copies out one case. **Peak memory O(one case)**, never O(N). Case phantom
is validated on read.

## 7. Metrics (arrays `(C,3,H,W)`, float64 reductions, dimensionless, no normalisation)
- `s1 = (1/C) sum_c sqrt(sum_{m,x,y}(t-r)^2 / (3HW))`.
- `s2 = max_{c,ROI} sqrt(sum_{m,roi^2}(t-r)^2/(3 roi^2))`, 25x25, stride 1, skip ROI if truth max (all materials) <= 0.01.
  **Default replicates the kit loop `range(H-roi)`, which omits the last position**; `include_last_position=True`
  gives the complete scan. Published s1 = 6.8e-7 is not a noise floor.
- Per-material MAE/RMSE, pooled RMSE (differs from s1), `partition_errors` (|Sp-1| on truth-tissue, |Sp| on
  truth-air, reported separately, NaN if the regime is empty), `boundary_rmse` over truth-band pixels.
