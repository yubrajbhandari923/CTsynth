# AAPM DL-Spectral benchmark: which comparisons are meaningful (methods review, read-only)

Evidence: repo notes only (hellwege2026, peng2024, valand2026, aapm-dl-spectral-ct.md, physics spec, EXP-002/F-010).
AAPM facts below are from search excerpts (starting_kit NOT opened). Tags: [V] verified in repo notes, [S] secondary excerpt, [U] unverified.

## 0. Mismatch to settle before anything else
- Our `ctsynth.physics` is 2D PARALLEL beam, exact chord projector, mass-density basis, dense chord matrix O(views*dets*pixels).
- AAPM data are circular FAN-beam, 512x512 images, 256 views x 1024 dets per kVp, interlaced fast-kV (low/high use different view angles) [S]. 50/80 kVp [S].
- Consequence: we cannot forward-project AAPM-geometry data with the current projector. Raw-sinogram methods (d,e,g-with-sinogram-loss) need either a fan-beam projector (new, tested code) or are restricted to the provided FBP images. Dense 512^2 x 256 x 1024 chord matrix is ~1e11 entries: impossible even sparse-free; needs matrix-free/GPU.
- Spectra, detector model, noise model, distances, units [U] until the starting kit is read. Every polychromatic method depends on them.

## 1. Method classes
| Class | Needs | Tells us | Feasibility |
|---|---|---|---|
| (a) Image-domain linear 3-material from 2 FBP images (+ sum-to-one) | the two FBP images, 3x3 mixing matrix (calibrate from truth maps or from ROIs, as Peng) | Floor for "just invert"; how much beam hardening/noise breaks linear mixing (FBP of -log polychromatic data is NOT linear in f). | NOW (per-pixel 3x3 solve, seconds). Calibrating the matrix on training maps is legitimate only if train-split-only. |
| (b) Constrained LS / likelihood, nonneg + sum/partition | (a) plus a per-pixel NNLS/QP; or per-ray Poisson MLE (ours) with f>=0, sum f = 1 | Whether constraints alone resolve the ambiguity; the "classical baseline" | Image-domain NNLS: NOW. Per-ray MLE on AAPM sinograms: needs fan-beam geometry + known spectra + noise level [U]; small subset NOW, full set HPC. |
| (c) TV-regularised | (b) + TV prior, weight lambda tuned on validation | Noise/edge trade-off; Peng's TV baseline lowers means (L1 shrinkage) - expect it to hurt small calcifications | Image-domain TV (Chambolle/ADMM, CPU): NOW. Joint with projector (MBIR+TV): HPC. |
| (d) Sinogram decomposition then FBP | line integrals A_m per ray from 2 measurements (+ per-ray constraint, see sec 2), fan-beam FBP | Removes beam-hardening bias that (a) suffers; noise amplified into streaks | Needs matched rays for low/high (interlaced views -> interpolation in angle, itself an error source). Tiny synthetic parallel-beam NOW; real data needs fan-beam FBP + decisions on interlacing; modest compute once code exists. |
| (e) One-step MBIR (Hellwege's MBIR lambda=0 is the template) | differentiable polychromatic fan-beam projector, spectra, init, 10^2-10^3 iterations of 512^2 projections | Best-physics classical reference; sets what the physics can deliver | Tiny parallel-beam 32-64 px NOW (CPU/MPS). 512^2 fan-beam: HPC GPU (needs matrix-free projector; not built). |
| (f) Supervised U-Net images -> maps | train split of images+truth, GPU hours; 1000 cases | Upper reference for data-driven; what the challenge winners did | 64-128 px subsampled: laptop MPS feasible but a different problem. Full 512^2: HPC GPU. Hellwege used UNet++ ResNeXt101 on A6000. |
| (g) Physics-consistency / unsupervised (Hellwege) | (f) architecture + differentiable polychromatic fan-beam projector + exact spectra; loss = reprojection MSE | Whether projection consistency alone trains a decomposer; here with 3 materials, whether it is identifiable at all (sec 2) | Needs (e) infrastructure and GPU: HPC. Hellwege is 2-basis, simulated-only, same model for data and training (inverse crime) [V]; their numbers are not comparable to a 3-material task. |

Not comparable: Peng2024 (iodine/bone, ROI SD metrics), Valand2026 (rho/Zeff, PCCT), Hellwege Table 1 (VMI RMSE in HU, 2 bases). The AAPM metric is RMSE of tissue maps, exact aggregation [U]. Published challenge numbers (near float32 precision for top teams [S]) are full-res 512^2 with the hidden test set; do not put ours beside them.

## 2. Identifiability: how can 3 maps come from 2 spectra?
F-010 claim: per-ray (or per-pixel) Fisher rank <= min(n_meas, k). For dual-kVp that is <= 2. Correct, and it applies to the UNCONSTRAINED mass-density parameterisation (k=3 free densities per point): at most a 2-D subspace of (adipose, fibro, calcification) is determined; a 1-D null direction remains. So with no extra information the 3-material problem is NOT identifiable pointwise. No contradiction with the AAPM task: the task supplies extra information, and the question is which.

Candidate extra information, assessed from first principles (mu = sum f_m mu_m at each energy, linear in f for the idealised model):
1. Sum-to-one / partition (sum f_m = 1 inside the breast): ADDS ONE LINEAR EQUATION. 2 spectral equations + 1 constraint = 3 equations, 3 unknowns -> unique solution iff the three material points (mu_m(low), mu_m(high)) are non-collinear (barycentric coordinates). THIS IS THE MINIMAL, SUFFICIENT FIX. The rank bound is not violated: the constrained model has k-1 = 2 free parameters, equal to n_meas. Stated for the AAPM phantom: "material maps sum to 1 inside breast, transition to air at edges" [S, excerpt of challenge page; Zenodo says maps are fractions]. Not verified numerically on the files (check `A+F+C` histogram on one case once downloaded). Edge pixels (partial volume with air) violate sum=1, so air must be a 4th "material" or a mask must be used.
2. Mutual exclusivity of calcification: NOT stated in sources [U]. As a prior (calcification pixels have f_C=1) it would help but is not a property we can claim. Physically, microcalcifications are small and partial-volumed, so exclusivity is false at edges.
3. Nonnegativity (f>=0): does not restore rank; it bounds the null direction to a segment (a polytope), reducing but not removing ambiguity. Useful as a regulariser, not as the identifier.
4. Spatial priors / sparsity / piecewise-constant: resolve the ambiguity only statistically (calcification sparse, FG regions smooth); can work but are priors, not measurements; learned methods (f) implicitly use exactly this. Phantom is stochastic with random FG structure and variable calcification count [S]: not piecewise-constant in a strict sense [U], inclusions are small.
5. Distinct spectral signature of calcification: necessary, not sufficient. Calcium differs strongly from soft tissue in 50/80 kVp; adipose vs fibroglandular are close, so the triangle is thin in that direction. This governs CONDITIONING, not rank. Expect adipose/FG separation to be noise limited even though calcification is well separated. Compute the 3x3 condition number from the starting-kit tables before expecting anything.

Verdict: the classical baseline CAN work, via (1) alone, provided the phantom really is a partition of unity (verify) and the mixing matrix is well conditioned (verify). It does not work without the constraint (cf. F-010). Caveats:
- The constraint is linear in volume fraction f, not mass density: use rho = f*rho_ref (our MaterialBasis convention) consistently; a sum-of-densities constraint is wrong.
- Per-ray (sinogram) form: sum_m A_m / rho_ref_m = chord length inside the breast along that ray. This needs the breast support per ray (from a first-pass reconstruction or mask), an extra approximate input for (d)/(e); error there maps into material bias. The image-domain/MBIR form (constraint per pixel) avoids this.
- Under noise, "well posed" is not "accurate": F-010 condition numbers (~200 for dual-kVp k=2) imply strong noise amplification; the constraint makes k_eff=2 with the AAPM tissue pair being closer than iodine/water, probably worse. Unquantified until spectra are known.
- Unsupervised (g): projection-consistency loss alone is blind to the null direction without the constraint; with 3 outputs it needs either the sum-to-one layer (softmax output) or it silently trains on prior/inductive bias. Do not compare to Hellwege's 2-basis result.

## 3. Minimal honest ladder
R0. Synthetic tiny, NOW (parallel-beam 32-64 px, our spectra, our model): verify (a)(b)(d)(e) recover known maps with the sum-to-one constraint, and FAIL without it (shows F-010 empirically, null direction). Sweep noise N, condition number. Could support: the identifiability mechanism and the estimators' implementation correctness. Could NOT support: any AAPM-performance claim, any beam-hardening claim vs real data (shared model = inverse crime).
R1. Synthetic with deliberate mismatch, NOW: generate with finer energy grid / extra scatter / different spectrum than the estimator assumes. Supports: robustness ranking (EXP-002 showed bias always appears). Cannot support: absolute accuracy.
R2. Real AAPM data subset, NOW (download needs explicit approval; 3.4 GB total, FBP images + truth are the lighter ~1.5 GB; start with ~20-50 cases): (a),(b)-image-domain, (c)-image-domain on the provided FBP images, mixing matrix fit on a train split only, metrics on held-out cases. Supports: statements about image-domain baselines at stated resolution (full 512^2 is cheap per-pixel). Cannot support: anything about the sinogram methods or our physics model.
R3. Real data sinogram methods, needs new fan-beam projector + starting-kit spectra (small dev task, then NOW at reduced size / few slices, HPC for all): (d), (e). Supports: whether polychromatic modelling beats (a). Cannot support: comparison to challenge leaderboard unless full-res, same metric, same test protocol.
R4. HPC GPU: (f) supervised U-Net on train split with proper val/test; (g) physics-consistency. Supports: learning vs classical vs unsupervised comparison. Cannot support: generalisation beyond this phantom family and simulator (single hypothetical system, 2D).

## 4. Traps (any one makes the benchmark worthless)
- Resizing sinograms/images (e.g. 512->128) without changing detector pitch, pixel size and view count in the geometry: the physics no longer matches the data. Downsample by simulating at low-res from the truth, or rebin consistently and recompute geometry.
- Inverse crime: simulating data with the estimator's own model (our R0 is best case; Hellwege shares this). Always mismatch the generator.
- Splitting by slice/crop instead of by phantom case; tuning lambda, mixing matrix, early stopping, or network on test; using hidden-test numbers we do not have (test truth not in the Zenodo record [S]). Carve train/val/test by case ID once, freeze test.
- Calibrating the mixing matrix or thresholds on the same truth used for scoring.
- Comparing laptop low-res/subset numbers to challenge or Hellwege numbers (different res, set, metric, basis, units).
- Ignoring that sinograms are interlaced (each kVp has different view angles) when pairing low/high rays.
- Forgetting air/edge pixels: sum-to-one fails at breast boundary; metrics over the whole image or only inside the breast give different answers - state the mask.
- Treating "constraint makes it identifiable" as "accurate": always report condition number and per-tissue RMSE (esp. adipose vs FG, calcification separately), not only mean RMSE.
- Mixing mass-density and volume-fraction units (spec forbids passing fractions to the model).
- Reporting a single noise realisation or seed; claims from one case.
