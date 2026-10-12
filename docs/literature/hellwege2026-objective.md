# Hellwege et al. 2026 (arXiv:2604.25356v1) - objective and assumptions audit

Source read in full: arXiv PDF (https://arxiv.org/pdf/2604.25356), 12 pp; abs page 2026-04-28. Section/eq numbers are the paper's.
Figures not viewable (captions only). No v2 seen. Marks: VERIFIED = read in the text; UNVERIFIED = not stated.

## 1. Loss (Eq. 6) - VERIFIED
Text: "trained to minimize the Mean-Squared-Error (MSE) between measured and predicted projections for both energy spectra":
  L(theta) = (1/|D|) sum_{(pLE,pHE) in D} (1/M) sum_{i=1..M} [ ||pLE_i - phatLE_i||_2^2 + ||pHE_i - phatHE_i||_2^2 ]
- Domain: LOG-TRANSFORMED line integrals. Eq.1: p_S,i = -ln (P_S(A f))_i, with Eq.2 normalised intensity inside the log. Not transmission counts, not images, not Poisson.
- Plain unweighted MSE: no per-ray weights, no photon-statistics weighting, no I0 (S is "normalized source spectrum including detector response", so p is dimensionless; Fig.3: "relative intensities (unitless)").
- Normalisation: 1/|D| per dataset sample, 1/M inside. Index i is never defined for Eq. 6 (Eq. 7 says M = "total number of projection measurements across both energy spectra", but Eq. 6 sums LE and HE terms inside one 1/M sum, so whether M counts rays, rays x views, or both spectra is AMBIGUOUS). Batch = 8; loss per-batch scaling not stated.
- No regulariser term. "unregularized DL approach" (Sec 2.2). No GT term.

## 2. Forward model inside the loss - VERIFIED
- Polychromatic (Eq. 2): p = -ln sum_k S(E_k) exp(-sum_X mu_X(E_k) sum_j a_ij f_Xj). Spectra assumed known: "requires only ... knowledge of the X-ray spectra, detector response and projection geometry".
- Number of energy bins E_k: UNVERIFIED (not stated). Spectrum exactly known: yes by construction; data generated with the same model (see 7).
- Noise: NOT modelled in the loss. Whether the AAPM challenge data used have noise is UNVERIFIED in this paper (F-012 notes the challenge data as noiseless). Introduction says the problem "amplifies noise" but no noise-level study exists. Only noise statement is none.
- Eq. 3 is stated as mu_Xj(E_k) = f_Aj mu_A(E_k) + f_Bj mu_B(E_k), where f are "linear attenuation contributions"; units/normalisation of f (volume fraction vs. coefficient) UNVERIFIED.

## 3. Materials vs measurements - VERIFIED, critical
- TWO basis materials, TWO spectra (LE, HE). Eq. 3 "two basis materials A and B"; Sec 2.4: "The materials soft tissue and calcification were chosen as basis materials". Network has "two input and output channels".
- Per pixel: 2 unknowns (fA,fB), 2 measurements (pLE,pHE) -> square, determined by counting (modulo the nonlinear polychromatic map and the line-integral/tomographic coupling). It is NOT 3-from-2.
- The AAPM dataset itself has 3 tissue classes (soft tissue/adipose+fibroglandular, bone, calcification; paper says "soft tissue, bone and calcifications"). The authors collapse to 2. How fat and fibroglandular were merged into one "soft tissue" GT, and how the GT VMI was built (Eq. 8 uses the 2-basis mu_A, mu_B?), is UNVERIFIED. If GT VMIs came from true 3-material maps, a 2-basis model cannot match them exactly; the paper does not say.
- => Their result says NOTHING about our 3-material-from-2-spectra case (F-012/F-014 null direction).

## 4. Extra constraints / priors - VERIFIED (all absent explicitly)
- Nonnegativity: none stated. Sum-to-one: none (and incompatible with their coefficient f definition anyway). Bounds, TV, entropy: none. Regulariser: none (lambda = 0 for MBIR too, "fair comparison").
- Implicit priors, none ablated: (a) UNet++/ResNeXt-101 architecture (CNN image prior); (b) BP inputs; (c) amortisation across 800 training sets (shared weights); (d) progressive 64/200/1000/full training and cyclic LR, said "proved beneficial" (no data shown); (e) train-only augmentation (rotations, flips, elastic). 
- ABLATIONS: NONE in the paper (4 figs, 1 table; no ablation of architecture, augmentation, progressive schedule, or any prior).
- Only comparison: DL vs EDEC, AEDEC, MBIR(lambda=0). MBIR minimises the same Eq. 5 objective yet scores 2x worse in VMI RMSE (Table 1: e.g. 50 keV 13.07 vs 5.83 HU); the difference is attributable to the network/training procedure, i.e. an implicit prior, not to the loss.

## 5. Network - VERIFIED / partly UNVERIFIED
- UNet++ [16], ResNeXt-101 (32x16d) encoder [17], Segmentation Models PyTorch. Input: 2-channel simple backprojection of pLE, pHE (f~LE, f~HE). Output: 2 channels (fA,fB).
- Output parametrisation: final activation NOT stated -> UNVERIFIED. Nothing indicates softmax/sigmoid; with 2 outputs and unconstrained "linear attenuation contributions" a softmax sum-to-one is unlikely, but unconfirmed. No architectural sum-to-one constraint is claimed.

## 6. Training protocol - VERIFIED
- Dataset-level (amortised) training, not test-time per-case optimisation: "Training is performed in batches of all training data"; 1000 sets, 80/20 split, 20% of train held for validation, 200 test. Adam, base lr 1e-4, cyclic restart at 1e-3 decaying over 10 epochs, batch 8, early stopping patience 150 epochs on validation loss (the projection-domain loss; no GT needed), best-val model kept.
- Hyperparameter selection: how lr, patience, subset sizes, encoder were chosen is UNVERIFIED (not described). Use of GT material maps for model selection: not stated; the described stopping uses the unsupervised validation loss. Test-set GT VMIs used for evaluation only (Sec 2.5). No statement that the test set was not used for tuning beyond this.
- Reported stats: Table 1 caption "Mean", Sec 3.2 text "median"; "+-" undefined (IQR? SD?). Fig. 4 example is deliberately a relatively poor case (6.10 HU).

## 7. Inverse crime - VERIFIED quote (Sec 4.2)
"It should be noted that the simulation of measured data and reconstruction have been performed with the same polychromatic model, which poses an significant advantage over typical settings, where the model has to be estimated."
- Consequences if models differed (our inference, not tested by them): the zero-residual solution set no longer contains truth; with a near-degenerate material pair the data-consistent set is large, so a mismatch would be absorbed along poorly-determined directions (cf. F-012: ~1000x amplification with the sum row). No spectrum, detector, or noise mismatch experiment is reported.

## 8. Code availability - VERIFIED absent
No code/repo link, no supplementary; "Data" statement only says AAPM dataset is public. Prior work [14] is "submitted, 2024". Generative AI was used in manuscript preparation.

## Identifiability (answer to F-014)
- Paper never discusses identifiability or null spaces. It is made well posed only by COUNTING: 2 materials from 2 spectra, per-pixel square system (their ill-posedness discussion is purely noise amplification). Not by an explicit constraint, not by softmax, not by a stated spatial prior.
- Their reported accuracy (without ablation) therefore additionally relies on unanalysed implicit priors: CNN architecture, amortisation across 800 cases, augmentation, and noiseless same-model simulation. With 3 materials the F-014 mechanism (|e.mu^T| < 1e-12) would leave the loss exactly flat along the adipose/fibroglandular null direction; the L(theta) in Eq. 6 contains nothing that selects a point on it. Any pinning there would come from the network prior or the dataset, not the physics loss, and the paper does not test that.
- Conclusion: Hellwege 2026 is not evidence that a pure projection-MSE objective identifies 3 materials from 2 spectra.

## Could not verify
Energy-bin count, final activation, units of f, 3-to-2 material merging and GT VMI construction, whether AAPM data were noisy, M definition in Eq. 6, hyperparameter-selection data, Fig. 2-4 contents, mean vs median, code.
