# Hellwege et al. 2026 - Unsupervised physics-informed DL for DECT material decomposition

## Bibliographic record
- Title: Unsupervised Physics-Informed Deep Learning for Dual-Energy CT Material Decomposition
- Authors: Laura Hellwege, Johann Christopher Engster, Moritz Schaar, Thorsten M. Buzug, Maik Stille
  (Univ. Luebeck; Fraunhofer IMTE)
- Year/venue: 2026 (v1 dated 28/29 Apr 2026), arXiv preprint, arXiv:2604.25356v1 [physics.med-ph]. No DOI; not peer reviewed as far as seen.
- Verified: reference-lookup `lookup_reference` on arXiv:2604.25356 returned the same title/authors/year; the PDF text
  header (https://arxiv.org/pdf/2604.25356) matches.

## Access level
Full text read (arXiv PDF, all 12 pages incl. refs). Figures were not viewable; only captions and extracted text.

## Problem
DECT material decomposition is ill-posed and noise-amplifying. Supervised DL needs ground-truth material images, which
cannot be measured clinically (basis coefficients are abstract; only derived quantities such as VMIs can be checked
against physical references, Sec. 1, 4.1).

## Method (Sec. 2)
- Inputs: measured LE/HE sinograms, backprojected (simple BP) to two image-domain channels. Outputs: 2 basis-material images.
- Forward model (Eq. 1-3): polychromatic, p_S = -ln sum_k S(E_k) exp(-sum_X mu_X(E_k) sum_j a_ij f_Xj); two-basis
  linear model mu(E) = f_A mu_A(E) + f_B mu_B(E). S includes source spectrum and detector response. Needs known spectra,
  detector response, geometry.
- Net: UNet++ with ResNeXt-101 (32x16d) encoder (Segmentation Models PyTorch), 2 in / 2 out channels.
- Loss (Eq. 6): MSE between measured and re-projected (through differentiable polychromatic model) LE+HE sinograms. No
  regularizer, no GT material images. Gradient backpropagated through the physics model.
- Training: Adam, lr 1e-4 base with cyclic 1e-3 restarts (10 epochs), batch 8, progressive data subsets 64/200/1000/full,
  early stop patience 150, RTX A6000.
- Data (Sec. 2.4): simulated AAPM DL-Spectral CT Challenge, 1000 sets (80% train of which 20% val; 20% = 200 test).
  Basis = soft tissue + calcification. Train-only augmentation (5 random rotations/flips/elastic deformations applied
  in image domain before projection simulation). Simulation: SIMULATED ONLY.
- Baselines: EDEC, advanced EDEC (AEDEC, authors' own prior work), MBIR with lambda=0 (Eq. 5).

## Reported results
- Fig. 2: projection-domain RMSE over 200 test sets; DL lowest, "few outliers"; ordering EDEC > AEDEC > MBIR > DL
  (read from text, figure not seen). Fig. 3 examples: RMSE 1.29e-3 (good), 1.13e-2 (poor), unitless relative intensity.
- Table 1 (VMI vs GT, RMSE in HU / SSIM, 30/50/70 keV; value +- spread):
  EDEC 49.43/0.958, 24.75/0.967, 23.30/0.903; AEDEC 27.59/0.983, 18.14/0.985, 18.51/0.966;
  MBIR 31.24/0.977, 13.07/0.988, 13.32/0.981; DL 12.64/0.997, 5.83/0.999, 5.88/0.997.
- Fig. 4: DL still deviates most in dense regions (calcifications); example worst-ish case RMSE 6.10 HU at 50 keV.
- Internal inconsistency: Table 1 caption says "Mean", Sec. 3.2 text says "median". Which is correct is UNVERIFIED.

## Limits (authors, Sec. 4.2, plus reader notes)
- Authors: simulated data only; data simulated and "reconstructed" with the SAME polychromatic model, a "significant
  advantage" over practice where model must be estimated; fixed geometry (retrain per scanner); 2D slice-wise only.
- Reader notes: GT material images exist only because data are synthetic; claim of "no GT" training is shown but the
  evaluation still uses GT VMIs. No noise-level / dose study, no real scanner, no spectrum-mismatch test. No code link seen.
  Basis choice (soft tissue/calcification) fixed; no explicit statistics beyond spread in Table 1. Generative AI used
  for manuscript preparation (Acknowledgements).

## Relevance to CTsynth
- Directly supports a self-consistent loop: material maps -> polychromatic forward model -> sinograms. Eq. 1-3 is a
  ready template for an editable virtual-patient forward simulator and for a physics-consistency loss for inference.
- Shows that projection-domain consistency alone can train a decomposer, so synthetic virtual patients could supply
  inputs without needing paired labels, but their GT material maps are still the only way to validate basis coefficients.
- Warns of the inverse crime: identical forward model for data and training flatters results. A virtual-patient
  population should deliberately mismatch spectra/detector/noise between generation and inference to test robustness.
- Two-basis (not >2 materials) and basis-dependent coefficients: relevant to defining what "material truth" means.
- Dataset (AAPM DL-Spectral CT Challenge) is a candidate benchmark; its availability/licence beyond "publicly
  available through the AAPM" is UNVERIFIED.

## Verified vs assumed
Verified (read in text): everything under Method, Table 1 numbers, limitations quoted above, bibliographic record.
UNVERIFIED / not read: figure contents (Fig. 2-4 visuals), the AAPM challenge dataset details (spectra, kVp, noise
model), cited prior work [14] (BMC Med Imaging, "submitted"), AEDEC paper [7], peer-review status/journal version,
code availability, whether Table 1 is mean or median.
