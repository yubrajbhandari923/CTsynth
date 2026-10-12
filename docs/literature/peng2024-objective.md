# Peng et al. 2024 (Med Phys 51(9):6185-6195, doi 10.1002/mp.17255) - objective and assumptions audit

Sources this time: arXiv 2311.10641 PDF (v1 preprint, equations partly legible, Methods/Implementation text) and the PMC
page PMC11489026 (equations recovered in reconstructed form by the fetch tool; surrounding text quoted). Europe PMC
fullTextXML returned HTTP 500; Wiley page, supplement NOT obtained. Caveat: arXiv v1 also contains an unsupervised DDPM
branch and hyperparameters (e.g. lambda) that may differ from the published version; flagged where used.

## 1. Loss (VERIFIED for form; weight value from preprint only)
- Eq. 7 (PMC, reconstructed): Loss_DF = || A [x1; x2] - [mu_H; mu_L] ||_2^2, with [x1;x2] = G([mu_H; mu_L]) (Eq. 8).
  Domain: IMAGE domain, per-pixel linear attenuation coefficients mu (not HU, "to preserve the system's linearity", Sec 2.1),
  stacked over all N pixels (A is 2N x 2N, Eq. 3). Squared L2 over images; no stated per-pixel normalisation. Not
  sinogram, not counts, no Poisson weighting.
- Eq. 9: Loss_total = lambda * Loss_DF + Loss_GAN. "lambda is the user-defined weighting factor".
- Adversarial term: "conventional GAN loss"; PatchGAN discriminator (preprint Sec 3.5); exact GAN form (vanilla/LS) UNVERIFIED.
- Regularisers: none explicit (TV only in the baseline, Eq. 6). Noise suppression comes only from the discriminator.
- lambda: preprint v1 Sec 3.5: 1e4 (simulation), 1e2 (clinical). Published value UNVERIFIED (main text says
  "parameter settings in supplemental materials"). Generator: residual U-Net; Adam lr 1e-5, betas (0.5,0.999), batch 2.
- Quote (preprint): "the correlation between the generated material images and acquired DECT images is only enforced by the
  DECT data-fidelity loss, and the noise suppression is enforced through the generative-adversarial mechanism."

## 2. Supervised or not (VERIFIED)
- No paired material-map labels in the loss. Paper's own framing: unsupervised (no paired DECT / ground-truth maps).
- BUT the discriminator needs a REFERENCE set of "low-noise material-specific images" (unpaired). In the preprint these were
  made "from the high-energy CT images using the threshold method" ("not the ground truth, but only provide the feature of
  low-noise images") from OTHER patients (1000 slices from 12 brain / 15 lung patients). So the target distribution is a
  learned image prior built from thresholded-SECT 2-material maps. This is adversarial-with-unpaired-data, not
  label-free in the strict sense: it needs a curated, anatomy-matched set of clean material images (labels-in-distribution).
- Architecture is a pix2pix-like conditional GAN with L1 replaced by the data-fidelity term (authors' own comparison).

## 3. Physics (VERIFIED - note's claim confirmed, with a refinement)
- Linear two-basis image-domain model: [mu_H; mu_L] = A [x1; x2], A entries = mean mu of two manually chosen uniform ROIs
  (Eq. 1-3). "disregarding the polychromatic nature of X-ray beams ... sacrificing the ability to correct beam-hardening."
- The forward model IS in the loss, but it is a 2x2 per-pixel linear matrix applied after reconstruction. No spectrum, no
  projector, no Poisson likelihood. Authors say a nonlinear projection-domain fidelity "can" replace it (future work).
- Consequence: since A is square and invertible, Loss_DF = 0 has the unique solution x = A^-1 mu (the noisy direct inversion).
  The fidelity term therefore pulls G toward direct inversion; the GAN term trades that off for noise. Fidelity carries no
  information beyond what matrix inversion has, so no null-space/ill-posedness issue arises in their setting.

## 4. Counts (VERIFIED)
2 basis materials (bone/soft tissue; or bone/iodine, iodine/water in phantom) from 2 measurements (high/low kVp images,
or TwinBeam-type). Determined (2x2), no volume-conservation/sum-to-one row. Basis fixed at training (Sec 5: retrain per basis).
Contrast to CTsynth F-012/F-014 (3 materials from 2 measurements + sum-to-one): Peng does NOT address an underdetermined
system. Cannot be applied to the 3-from-2 AAPM problem without a new constraint.

## 5. Ground truth and evaluation (VERIFIED from preprint text; details in supplement UNVERIFIED)
- Simulation: CQ500 head, TCIA LungCT-Diagnosis; GT material maps from SECT by threshold + normalisation; DECT forward-
  simulated (preprint: Poisson noise with incident photons 1e4 per detector pixel, FBP to 256x256, 1 mm pixels). The
  evaluation GT and the training reference set come from the same threshold recipe (different patients): in-distribution,
  and a two-material threshold truth is crude. Simulated DECT is then inverted with a linear A (ROI-derived) - likely a
  partial inverse crime for the image-domain model; residual errors attributed to beam hardening/reconstruction.
- Physical phantom: Varian CBCT 140/100 kVp, Gammex iodine inserts, trained on 1500 simulated polygon phantoms with held-out
  iodine concentrations. Clinical head/lung: no GT, ROI SD only.
- Hyperparameter selection / held-out protocol: UNVERIFIED. Per-dataset lambda (1e4 vs 1e2) and per-dataset TV weights
  suggest tuning per task; no validation split or selection protocol is described in what I read.

## 6. Noise prior (VERIFIED mechanism; magnitude as reported)
Discriminator D is trained to separate reference low-noise material images from G's (noisy-input) outputs - a learned
image-manifold prior, "similar to image-prior regularization in iterative methods". It buys SD reduction vs direct
inversion: >97% (brain sim), >91% (lung sim), >93-95% clinical; SSIM vs GT ~0.2 -> >0.9 (preprint wording). Noise-only
metric is inflated by smoothing/hallucination; accuracy claim rests on ROI means "almost the same as matrix inversion".
No hallucination or uncertainty analysis seen.

## 7. Code (UNVERIFIED)
No code/data availability statement on the PMC page; search found no repository. Treat as not released (could not check
Wiley data-availability statement or supplement).

## KEY QUESTION: fair comparator for a label-free physics-consistency method?
No, not as a like-for-like label-free baseline. (a) Its fidelity term is a linear 2x2 image-domain identity with a unique
zero (the noisy inversion), so it has no polychromatic physics and no handling of under-determination; (b) its noise/
anatomy control comes from an adversarial prior fit to a curated set of clean thresholded material maps (needs label-like
data, only unpaired); (c) it is 2-from-2, so it does not address 3-from-2; (d) evaluation GT is a threshold-derived
2-material truth, and basis/lambda are fixed per task.
What it CAN be: (i) a learned-prior, image-domain, 2-material reference for DECT noise suppression where the question is
"unpaired-clean-reference prior vs physics-only prior"; (ii) a weakly supervised upper reference if CTsynth gives it
reference material maps; (iii) a precedent for held-out-concentration phantom validation and for reporting SD vs mean
bias separately. If used as a comparator it must be re-implemented (no code), on a 2-material subproblem, with the
reference set and lambda disclosed, and be labelled "unpaired-reference adversarial", not "unsupervised/label-free".
