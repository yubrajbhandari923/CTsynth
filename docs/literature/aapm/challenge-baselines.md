# 2022 AAPM DL Spectral CT Grand Challenge: published results and baselines

Sources: [S1] Sidky & Pan, Med Phys 51(2):772-785, DOI 10.1002/mp.16363; preprint arXiv:2212.06718 (v2), read via ar5iv HTML.
[S2] https://www.aapm.org/GrandChallenge/DL-spectral-CT/  [S3] Zenodo 10.5281/zenodo.14262737.
NOTE: the arXiv id 2303.09750 in the task brief is WRONG (it is an unrelated sensor-placement paper). Correct id: 2212.06718.
CAVEAT: S1 text/tables were extracted through an LLM page summariser (ar5iv); the PDF was not parsed locally (no pdftotext/pypdf).
Numbers below are therefore "VERIFIED (ar5iv, machine-extracted)"; spot-check Table 1 against the PDF before citing.

## 1. Design
- Single task (no sub-tasks): estimate 3 tissue maps (adipose, fibroglandular, calcification; volume fractions in [0,1], 512x512) of a 2D stochastic breast phantom. VERIFIED [S1,S2]
- Scan: fast kVp-switching 50/80 kVp, 256 views per kVp, 1024-px detector, 512x512 grid over (18 cm)^2. VERIFIED [S1,S3]
- Provided: 1000 training cases (transmission data, FBP images, truth maps), full forward model/spectra/Python code, 10 validation cases (no truth, unlimited leaderboard), 100 test cases (no truth, 3 submissions). VERIFIED [S1,S2]
- Dates: train release Mar 17, validation from Mar 31, test May 17-31 2022. VERIFIED [S1]
- Test ground truth: withheld in the report; no release statement found. Zenodo 14262737 lists training data only. 18 teams, 39 test submissions. VERIFIED [S1,S3]; later release elsewhere UNVERIFIED.

## 2. Inputs (important)
- BOTH allowed: transmission data (I_low, I_high; 256x1024 each), FBP images (negative log + FBP, 512x512), or both; image-to-image, data-to-image, hybrid, iterative. VERIFIED [S1,S3]
- Headline numbers are NOT tied to one input: each team used its own pipeline. GM_CNU, iTORCH, MIR etc. used model-based/iterative steps on transmission data (iTORCH is pure CG on transmission data, no CNN); DCPT+Navrit used FBP images only (U-Nets on low/high/difference images). VERIFIED [S1] (per-team input details beyond this UNVERIFIED)

## 3. Metric
- Ranking metric s1 = (1/100) sum_{i=1..100} sqrt( ||t_i - r_i||_2^2 / (3n) ), n = 512^2, t,r = triplets of tissue maps. Pooled over 3 materials with equal weight (no per-material weighting); RMSE per case then mean over cases. VERIFIED [S1]; S2 says "mean RMSE", tie-break worst-case 25x25 ROI-RMSE.
- s2 = max_{i,c} sqrt( b_c^T |t_i - r_i|^2 / (3m) ), m = 625, b_c = indicator of 25x25 ROI at c (same ROI location across maps). VERIFIED [S1]
- Units: dimensionless volume-fraction (fill fraction, range 0-1); report gives no explicit unit. cm^-1 appears only for the DE images in Fig. 4. VERIFIED [S1]

## 4. Results (S1 Table 1, top 10; s1 = mean RMSE, s2 = worst-case ROI-RMSE; dimensionless fractions)
| Rank | Team | s1 | s2 |
|---|---|---|---|
| 1 | GenwaiMa/GM_CNU | 6.80e-7 | 2.20e-6 |
| 2 | huxiaoyu090/iTORCH | 6.21e-6 | 9.77e-5 |
| 3 | kimhs369/MIR | 1.87e-4 | 6.81e-3 |
| 4 | WashUDEAM | 2.52e-4 | 2.44e-2 |
| 5 | dhlee91 | 4.08e-4 | 2.54e-2 |
| 6 | jaspernijkamp/DCPT+Navrit | 1.34e-3 | 4.60e-2 |
| 7 | Duke_QIAL | 4.01e-3 | 5.21e-2 |
| 8 | leekunpeng/BME_NUC | 4.10e-3 | 7.18e-2 |
| 9 | flutexu | 7.73e-3 | 8.24e-2 |
| 10 | Z-VCT | 1.04e-2 | 1.09e-1 |
- Figs 6-7: s1 vs s2 scatter for all 18 teams (Fig 7 on -log10 scale). Max pixel error: iTORCH 1.77e-3, GM_CNU 6.14e-6. VERIFIED [S1]
- Baselines: NONE reported (no FBP, one-step inversion, or MBIR reference score). We could not find any published baseline numbers. VERIFIED absence in the extraction; UNVERIFIED against full PDF.
- Context: 2021 sparse-view challenge winner mean RMSE 6.4e-6 (different problem, 10,000-case test set). [S1]

## 5. Top methods and code
1. GM_CNU (Ma, Zhao): JLRM-SCT; E-ART pre-decomposition + RED-CNN + 2 iterations of physics-model network + residual CNN, small kernels to limit overfitting.
2. iTORCH (Hu, Jia): no CNN; constrained conjugate gradients with training-derived constraints (sum-to-one, binary non-edge pixels, calcification sparsity); outer iterations [10,30,30], 200 CG each.
3. MIR (Kim, Cho): four U-Nets + linearised SART refinement.
4. WashUDEAM: DEAM alternating minimisation then U-Net on six basis maps.
5. dhlee91: unrolled primal-dual hybrid gradient network.
6. DCPT+Navrit: U-Nets on low/high/difference images, transfer learning from ideal VMIs.
7-10: Duke_QIAL (4-stage), BME_NUC (iDD-Net), flutexu (iterative + U-Net), Z-VCT (SART-NLM + 2 CNNs + algebraic refinement).
- Code: report does not state top-ten code is public (UNVERIFIED either way; check team GitHub pages by username).

## 6. Noise / solvability (changes what "good" means)
- Data are NOISELESS: "no additional inconsistency, such as noise, is introduced into the transmission data." Participants had exact knowledge of the forward model. VERIFIED [S1]
- Exact recovery "in principle possible"; proof of solvability seems unlikely; no noise floor discussed. VERIFIED [S1]
- Consequence: RMSE 1e-6 to 1e-7 is reachable (limited by float32/optimisation, not noise). Values of 1e-2 to 1e-3 are mid/low-pack. Do not read these as clinically meaningful accuracy; they measure how well the inverse is solved on noiseless, matched data. (our inference)
- Float32 data storage (S3) implies a precision floor near 1e-7 relative; GM_CNU 6.8e-7 is near it. (our inference)

## 7. Pitfalls
- Inverse crime: forward model, phantom family and generator identical between train and test and shared with all; report notes results may not generalise to other objects/configurations. VERIFIED [S1]
- Only 100 test cases; algorithms should be checked on larger sets. VERIFIED [S1]
- Overfitting: GM_CNU used small receptive fields; no formal leakage analysis. Validation leaderboard was visible; test scores were visible to all. VERIFIED [S1]

## Could not verify
- Table 1 against the PDF itself; any baseline score; code links; per-team input choice; any post-challenge test-truth release.
