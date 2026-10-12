# Label-free decomposition machinery (T-015)

Code: `src/ctsynth/aapm/{torch_forward,unet,losses,train}.py`; tests `tests/aapm/test_{torch_forward,unet,losses,train}.py`.
Synthetic data only. Context: F-018, F-019, F-020, D-006 (B3 is a NEGATIVE CONTROL, not a reproduction).

## Scope limit (read first)
There is **no fan-beam projector** in this repo (`ctsynth.physics` is parallel beam; AAPM is fan beam). The path-domain
model therefore operates on SUPPLIED material line integrals. **A fan-beam projector is required before any
sinogram-domain AAPM experiment; image-domain experiments are the only ones we can run today.**

## Forward (`AapmTorchForward`, float64-capable, gradcheck-tested)
Channels: 0 = 50 kVp, 1 = 80 kVp. Materials in `schema.MATERIALS` order.
- Image: `A[k,m] = sum_E 0.5 s_k mu_m` (cm^-1); `(N,3,H,W)` fractions -> `(N,2,H,W)`.
- Path: `T_k = sum_E 0.5 s_k exp(-sum_m mu_m P_m)`, `P` in cm, I0 = 1, noiseless; `(N,3,...)` -> `(N,2,...)`.

## Output parametrisations (`UNet(depth, width, output)`)
| output | constraint | note |
|---|---|---|
| linear | none | |
| softplus | >= 0 | nonnegativity alone does not identify (F-019) |
| softmax | >= 0, sum = 1 everywhere | imposes the identifying constraint everywhere; WRONG for air (~35%) and band (~5%) |
| softmax_with_air | >= 0, sum <= 1 | 4-way softmax, air channel dropped; air representable |
Default `UNet(2, 8)` = 29,411 params (depth 2, width 8); tests use depth 1-2, width 4 (1.7k-7.4k). H, W divisible by 2**depth.

## Loss terms (`losses.REGISTRY`, each unweighted; `compose` applies weights, weight 0 = not evaluated)
| term | definition | divisor / units |
|---|---|---|
| consistency_image | (A frac - meas)^2 | mean over N*2*H*W; (cm^-1)^2 |
| consistency_transmission | (T(paths) - meas)^2 | mean over N*2*R rays; dimensionless |
| nonneg | relu(-frac) | mean over N*3*H*W |
| sum_to_one | (sum_m frac - 1)^2, optional bool mask | mean over masked pixels |
| tv | sqrt(dx^2+dy^2+eps)-sqrt(eps), forward differences | mean over N*3*(H-1)*(W-1) |
| entropy | -sum p log p, p = relu(frac)/sum | mean over pixels, nats, [0, ln 3]; HYPOTHESIS, not a known-good prior |

## Training (`train.py`)
- `train_label_free(factory, fwd, meas_train, meas_val, weights, ...)`: signature contains no truth argument
  (structural guard; extra kwargs raise TypeError). sum_to_one mask is tissue = NOT(both measured images < air_cutoff),
  derived from measurements only. Image-domain only (no projector for transmission-domain training).
- Early stopping on validation `consistency_image` (no ground truth); best-validation weights restored.
- `train_supervised(...)`: separate function, labels explicit. Its `val_consistency` history key holds label MSE.
- Determinism: init under `fork_rng` + `manual_seed(seed)`; batch order from explicit `torch.Generator`.

## Key test (operationalises F-018/F-019)
Independent numpy Newton builds the F-018 decoy ([8,4,0.05] vs calc=0.10 -> [12.72,-0.51,0.10], ||dx|| = 6.53 cm).
`consistency_transmission` is 0 for truth and 4.8e-34 for the decoy: indistinguishable. For a nonnegative tissue decoy
([0.6,0.39,0.01] -> calc 0.013), `nonneg` is 0 for both, `sum_to_one` separates them. The F-018 big decoy has f = -0.51,
so `nonneg` does flag THAT one (bounds the family, does not eliminate it).
