# Classical constrained image-domain baseline (T-011)

Code: `src/ctsynth/aapm/classical.py`; tests: `tests/aapm/test_classical.py`.

**Validated on SYNTHETIC data only.** No AAPM case data exists locally; nothing here is an AAPM result.
Image-domain inversion is expected to be far worse than the published transmission-domain winners
(D-005, s1 = 6.8e-7). The inputs here are the two reconstructed images, not the transmission data.

## Model
Per pixel `b = A x`, `b = (img_50kVp, img_80kVp)`, `x = (adipose, fibroglandular, calcification)` volume fractions
(inferred units), `A[k,m] = sum_E 0.5 s_k(E) mu_m(E)` (F-012: `[[0.26697,0.31321,3.01182],[0.21930,0.24788,1.64238]]`,
reproduced by `effective_mu`). `A` is 2x3: null direction `[-0.723,+0.691,-0.008]`; with a sum-to-one row
singular values (3.697, 1.176, 0.00290), cond = 1273.

## API (images `(..., H, W)` -> fractions `(..., 3, H, W)`, float64)
- `decompose_lstsq(lo, hi, mu, *, mode, air_cutoff)`; modes `unconstrained` (pinv, min-norm; the demonstration
  that the error is exactly the null direction), `sum_to_one` (exact 3x3 solve), `simplex` (Euclidean projection of
  the sum_to_one solution onto `{x>=0, sum=1}`; not the weighted-LS optimum over the simplex). Vectorised.
- `decompose_tv(lo, hi, mu, *, lam, n_iter=300, air_cutoff, tv_eps, history)`: minimise
  `0.5 sum ||A x - b||^2 + lam TV_eps(x)` s.t. simplex on tissue pixels, 0 on air. Monotone FISTA, fixed step
  `1/L`, `L = sigma_max(A)^2 + 8 lam/tv_eps`, exact projection, Huber-smoothed isotropic TV
  (`tv_eps = 1e-2`), warm start at the closed-form simplex solution, deterministic. **`lam` is not tuned here and
  must be selected on the VALIDATION split only.**

## Air rule (F-013)
The phantom is two-regime: air (sum 0, ~35% of pixels, no fourth class), tissue (sum 1, ~60%), boundary
(0 < sum < 1, ~5%). A pixel is declared AIR iff **both** images are below `AIR_CUTOFF_PER_CM` (default 0.1 /cm =
half the smallest tissue effective mu, adipose at 80 kVp); air returns exact zeros in all modes. Everything else is
treated as tissue. The cutoff is a **modelling choice, not a fact from the data**. On real data it should be set on
the validation split (e.g. the valley between the air and tissue modes of the image histogram), never on test
data. `air_cutoff=None` disables the rule.
Consequence for boundary pixels: coverage below ~45% become air (fractions lost); above it the unit-sum row
conflicts with the measurement and the conflict is amplified by cond ~1273, giving errors far larger than
`1 - sum` (large negative values under `sum_to_one`; `simplex` clips them). This is a limitation of the rule.

## Test tolerances (each justified in the test file)
Exact recovery 1e-10 (float64, eps*cond ~ 3e-13); null-direction angle to own SVD null < 1e-6 deg (pinv error is
exactly `-(n.x) n`), to F-012's 3-digit vector < 0.2 deg; conditioning: error equals `sigma_max(B^-1[:, :2]) |dlt|`
to 1e-4 and relative amplification lies in `(cond/10, cond]`; simplex vs brute-force active-set projection 1e-12;
TV lower RMSE than lam=0 by >20%, objective non-increasing, lam=1e-12 matches lam=0 to 1e-6.

## Measured on synthetic data (reference values, not claims about AAPM)
- Null-direction angle: unconstrained error vs own null < 1e-6 deg; own null vs F-012 vector 0.024 deg.
- Conditioning: worst-direction relative amplification 181 = 0.14 x cond(1273): same order, below the bound
  (the exact sum row is not perturbed).
- TV (24x24 piecewise-constant phantom, Gaussian noise 1e-3 on both images, seed 1): RMSE 0.0785 (lam=0) ->
  0.0499 (lam=1e-4, 300 it): modest gain (36%). Sensitive to lam: lam=1e-3 gives 0.077 at 300 it and 0.34 at 1500 it
  (over-smoothing erases the adipose/fibroglandular contrast, which the data term barely defends). The
  lam for the test was picked by a coarse scan on that same synthetic problem, so the gain is optimistic.
