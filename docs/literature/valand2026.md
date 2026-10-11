# Valand et al. 2026 — physics-informed GAN for density / Zeff from PCCT

## Bibliographic record
- Title: Truth-based physics informed estimation of material composition in spectral CT in terms of density and effective atomic number
- Authors: Jainam H Valand, Mojtaba Zarei, Jayasai Rajagopal, Nicholas Felice, Joseph Y Cao, Kirti Magudia, Danielle E Kruse, Kevin R Kalisz, Ehsan Abadi, Ehsan Samei (Duke CVIT / Radiology)
- Venue: Phys. Med. Biol. 71(3), 035025. Published 13 Feb 2026 (received 10 Jun 2025, accepted 5 Feb 2026). DOI 10.1088/1361-6560/ae4284
- Verified by: IOPscience landing page fetched via doi.org (title, authors, abstract, dates, citation line); ORNL and CVIT publication lists agree. reference-lookup (OpenAlex) returned HTTP 429, so it did NOT verify this record.
- PMC copy exists: PMC13173581.

## Access level
- IOP full text is paywalled: abstract only from the publisher.
- PMC page was read through a WebFetch summarizer (a small model's extraction of the page, not raw text). Section/table numbers and values below come from that extraction. Treat them as secondary until checked against the PDF. The extraction itself reported garbled Table 1/2 headers.

## Problem
Spectral (PCCT) material decomposition by maximum likelihood is noise sensitive. Deep-learning decomposition lacks physics constraints and lacks ground truth. Goal: output density (rho) and effective atomic number (Zeff) maps, trained on simulated data with known truth.

## Method
- Data: simulated, not real, for training. DukeSim modelling Siemens NAEOTOM Alpha; vendor recon (ReconCT). 48 XCAT phantoms (32 train, 16 test); 25 M / 23 F, BMI 18.2-38.0, age 18-76 (Sec 2.1).
- Acquisition grid: 120 kV, pitch 0.8; CTDI 6 and 1.5 mGy; detector thresholds [20,50],[20,60],[20,70],[20,80] keV; iodine 1.47-2.73 mg I/ml; kernels Br36f, Br40f, Br56f (Sec 2.1).
- Ground truth: rho from ICRU (1989) reference densities; Zeff from the SIRZ method (Champley 2019) (Sec 2.1).
- Network: GAN, 3D conv + U-Net generator, 3-conv discriminator. Input 6 channels (2 thresholds x 3 adjacent slices); output 2 channels (rho, Zeff) (Sec 2.2).
- Physics term: a distillation ViT predicts 67 parameters (exponent n, two alpha, 64 gamma weights) for the model mu ~ rho*alpha(E,A)*Zeff^n + gamma(Zeff,E), gamma including Klein-Nishina and coherent terms (Eqs 1-4). Loss (Eq 8): 300 L1 + 50 style + 10 LSGAN + 10 L_Ph (Sec 2.2).
- Training: Adam, cosine warm-up, 4x A6000, flips/affine/noise/jitter augmentation (Sec 2.3).
- Baselines: same GAN without physics loss; analytical Schneider stoichiometric method (RED used as rho proxy).
- Clinical: 6 patients (5 venous, 1 arterial). Reader study: 4 radiologists, 30 slices, 5-point liver lesion conspicuity, VMIs vs 70 keV VMI + rho/Zeff (Sec 2.4).

## Reported results
- Abstract: with physics regularization NRMSE 1.29% / 0.68%, SSIM 0.99 / 0.99, PSNR 29.8 / 29.04 dB (rho / Zeff presumably; the order is from the summarizer, not the abstract). Max RMSE 5.45% on clinical data.
- Reader study (abstract): median conspicuity 4.52 vs 4.13; TOST equivalence within +/-0.5.
- Via extraction: Table 1 analytical method NRMSE 16.74% for electron density; Table 2 max RMSE 5.45% = blood Zeff, arterial phase, up to 14% for analytical; Table 3 kernel variation <1% except one case 2.5%; ICC poor (0.194 / 0.287).
- Inconsistencies flagged by the extraction: test phantoms 16 (Sec 2.1) vs 12 phantoms / 100 images (Sec 2.3); reader-study CIs differ between abstract and Sec 3. Not independently checked.

## Limits (authors, via extraction; the abstract itself states only the small-sample caveat)
Out-of-training materials degrade accuracy; routine acquisition/recon only; no physical anthropomorphic phantom; RED used as rho proxy without calibration; one clinical task; 6 clinical cases; small reader sample.
Additional observations (mine, from the above): truth is limited to the XCAT material table, so the network learns that table's rho/Zeff mapping; the 'clinical' RMSE is against what reference is UNVERIFIED.

## Relevance
- Same group's pipeline (XCAT + DukeSim, known per-voxel truth) is a direct template for editable virtual patients with exact material labels: sweeping dose, kernel, thresholds, and contrast concentration is the kind of factor grid an editable population needs.
- Material inference target here is (rho, Zeff), not basis-material fractions; the model mu ~ rho*alpha*Zeff^n is a power-law approximation, so composition beyond two scalars is not recovered.
- Truth tissue properties are fixed by the phantom (ICRU table); there is no tissue-composition variability within an organ, which is the gap a physics-grounded editable population would fill. Domain gap (sim to clinical) was tested on only 6 cases.

## Verified vs assumed
- VERIFIED (publisher page, read): title, authors, venue, dates, DOI, abstract content (32 train / 16 test / 6 clinical, GAN, thresholds, doses, kernels, metrics, reader result).
- SECONDARY (PMC via summarizer, not raw text): everything tagged Sec/Eq/Table above, including DukeSim/NAEOTOM, SIRZ, loss weights, 67-parameter ViT, baselines, limits list.
- UNVERIFIED: full-text wording, exact table contents, how rho is defined for contrast-enhanced voxels, spectra/detector model details, code or data availability.
