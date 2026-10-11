# Salinas et al. 2025 — DL-generated heterogeneous phantoms, in-silico CT

## Bibliographic record
- Title: In-silico CT simulations of deep learning generated heterogeneous phantoms
- Authors: C S Salinas, K Magudia, A Sangal, L Ren, W P Segars (Duke CVIT / Radiology; Univ. Maryland Radiation Oncology)
- Venue: Biomed. Phys. Eng. Express 11(4), 045027. Published 10 Jul 2025 (received 11 Mar 2025, accepted 30 Jun 2025). DOI 10.1088/2057-1976/ade9c9
- Verified by: IOPscience landing page fetched via doi.org (title, authors, abstract, dates, data-availability statement); Segars' Duke Pratt publication list corroborates citation. reference-lookup (OpenAlex) returned HTTP 429 and did NOT verify it.

## Access level
Abstract and metadata only. Full text is paywalled and no open copy (PMC/arXiv) was found. Methods, results tables, and limits beyond the abstract are NOT read. Do not infer them.

## Problem (from abstract)
Virtual phantoms emphasise geometric accuracy; realism needs intra-organ texture and material variation, since tissue is heterogeneous.

## Method (abstract only)
- Two 3D Double U-Net conditional GANs (3D DUC-GAN) generate sixteen unique textures covering torso organs.
- Trained on 378 CT image-segmentation pairs from a publicly available dataset (which one: UNVERIFIED); 18 pairs held out for testing.
- Homogeneous phantoms in, textured phantoms out; textured phantoms imaged with DukeSim (virtual CT platform).
- Simulation plus real CT for comparison; spectra, scanner model, dose, recon: UNVERIFIED (not in abstract).

## Reported results (abstract)
- Vs original CT: mean absolute difference 46.15 +/- 1.06 HU; SSIM 0.86 +/- 0.004; PSNR 28.62 +/- 0.14; maximum mean discrepancy 0.0016.
- Stated improvements of 27%, 5.9%, 6.2%, 28% on these metrics versus current homogeneous texture methods (the abstract does not say which metric maps to which percentage; the order is not assumed here).
- Virtual-CT scans of generated phantoms visually closer to true CT than the previous method.
- Data availability: data owned by a third party, not public; available on reasonable request.

## Limits
Not stated in the abstract; full-text limitations UNVERIFIED. From the abstract alone: texture is in HU/appearance space (mean abs difference of 46 HU remains); nothing indicates the texture carries elemental composition or multi-energy-consistent attenuation. Whether the textures are mapped to materials is UNVERIFIED.

## Relevance
- Same XCAT + DukeSim ecosystem as Valand 2026; supports the claim that homogeneous organ labels under-represent intra-organ variation, motivating editable intra-organ heterogeneity.
- For spectral/DECT material inference: if textures are image-domain HU only, they cannot supply per-voxel composition ground truth for decomposition. Whether this paper maps texture to materials is UNVERIFIED; check full text before citing it as a material-variation source.
- Related work by the same group (Chang 2020 MT-XCAT D-CGAN; Salinas 2025 SPIE 3D CTGAN) appeared in search results only as titles; contents unread.

## Verified vs assumed
- VERIFIED (publisher page, read): bibliographic fields, abstract, dates, data-availability statement.
- UNVERIFIED: everything about methods detail, dataset identity, tissue/material assignment, scanner and spectral settings, tables/figures, stated limitations.
