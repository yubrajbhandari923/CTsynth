# AAPM DL-Spectral CT Grand Challenge dataset (Zenodo 14262737)

Retrieved 2026-10-11. Sources read: Zenodo API JSON (`https://zenodo.org/api/records/14262737`), the challenge page
(https://www.aapm.org/GrandChallenge/DL-spectral-CT) and the report abstract/excerpts (PMC10509324, arXiv 2212.06718) via
search excerpts only. No data files downloaded. `starting_kit.tgz` (103 MB) was NOT opened, so the forward-model spec is UNVERIFIED in detail.

## Record metadata (VERIFIED from Zenodo API)
- Title: "Data repository for the 2022 AAPM deep-learning spectral CT grand challenge"
- Creator: Sidky, Emil. Contributors: Pan, Xiaochuan (ProjectLeader); AAPM Working Group on Grand Challenges (HostingInstitution)
- Published 2024-12-02 (record created 2024-12-11); single version, no v2. Community: aapmhq.
- DOI (this version): 10.5281/zenodo.14262737; concept DOI: 10.5281/zenodo.14262736
- License: CC-BY-4.0. Access: open (no login/request). Type: dataset.
- Supplement to: doi 10.1002/mp.16363 (Sidky & Pan, "Report on the AAPM deep-learning spectral CT Grand Challenge", Med Phys 51:772-785, 2024; arXiv 2212.06718).
- Funding (NIH): R01EB026282, R01EB023968, R21CA263660.
- Zenodo stats at retrieval: 2505 downloads / 1953 unique.

## Files (VERIFIED, API; md5 checksums)
| File | Bytes | md5 |
|---|---|---|
| highkVpTransmission.npy.gz | 842,546,719 | 57011ed1b0346d1062627a6ba86f6b7e |
| lowkVpTransmission.npy.gz | 850,758,740 | 8c48d814d3db4c34f1cb96348cd1c9d9 |
| highkVpImages.npy.gz | 716,205,150 | f600ba6546c0f8fab3b227e3d470c5ed |
| lowkVpImages.npy.gz | 717,374,937 | 8afab98fef99400d4f293f57281fa6a3 |
| Phantom_Adipose.npy.gz | 102,183,683 | 1d0c7239fa747a285c7ef1bcf684e3da |
| Phantom_Fibroglandular.npy.gz | 113,764,766 | 746641e2910c231a39441d242bdfd458 |
| Phantom_Calcification.npy.gz | 6,937,257 | 2e04dc154dc36087a2404007c1b36baa |
| starting_kit.tgz | 102,706,887 | 7aaf6b55708a6576ebfa91e43ec80082 |
Total about 3.45 GB compressed (sum of the above). Only starting_kit.tgz and Phantom_Calcification are near/under the 100 MB range; none <50 MB except Calcification.

## Data format and contents (VERIFIED from record description)
- NumPy `.npy`, float32, gzipped; load with `numpy.load` after gunzip.
- 1000 training cases. Phantom_{Adipose,Fibroglandular,Calcification}: 1000x512x512 ground-truth tissue maps (fractions).
- {low,high}kVpTransmission: 1000x256x1024 = 256 views x 1024-pixel linear detector, per kVp (interlaced fast-kV-switching, so each kVp has 256 of 512 total views).
- {low,high}kVpImages: 1000x512x512, from -log of transmission then FBP (the "standard processing" baseline; beam-hardening/streak artifacts present).
- Transmission values are described as "transmission" data (post-normalization intensity ratio, noise-free vs noisy: UNVERIFIED; the challenge page text I read did not state noise).
- Test set: 100 cases (report excerpt); NOT in this record (training-only). Hidden-test truth availability: UNVERIFIED (not in file list).

## Physics / forward model (from challenge page + report excerpts; details UNVERIFIED)
- 2D simulated breast CT; stochastic phantom: random fibroglandular structure in circular breast cross-section, variable number of microcalcification-like inclusions, skin line modelled as fibroglandular. Material maps sum to 1 inside breast (partial-volume at edges), transition to air outside.
- Circular fan-beam, 512 views over 360 deg, source spectrum alternates 50 kVp / 80 kVp (low/high). Challenge page gives these as 50 and 80 kVp.
- Three-material basis (adipose, fibroglandular, calcification) -> inverse problem is spectral reconstruction + decomposition.
- Full physical model is in starting_kit.tgz: polychromatic spectra, attenuation tables, detector model, geometry. NOT READ: spectra files, energy-bin count, detector response/noise model, source-detector distances, pixel/detector pitch are UNVERIFIED.
- Hypothetical system (not a real scanner); the "Reiser & Nishikawa" breast-tomosynthesis task-based paper is cited on the page for the phantom basis (UNVERIFIED how used).

## Tasks, metrics, splits
- Task: estimate the 3 tissue maps from transmission data, FBP images, or both (data-to-image, image-to-image, hybrid, or model-based iterative).
- Metric (report): RMSE of tissue maps vs truth (top teams reached near float32 precision; mean RMSE of order 1e-6 for the previous sparse-view challenge cited, not this one). Exact per-tissue weighting/aggregation formula: UNVERIFIED.
- Split: 1000 train (this record), 100 test (hidden at challenge time; report states 1,000 and 100). 18 teams submitted in the test phase.
- No in-record validation split; participants carved their own.

## Verified vs assumed
VERIFIED (read directly): title, creator, DOI, version/date, CC-BY-4.0, open access, file names/sizes/md5, array shapes, dtype, file format, task statement, FBP-image provenance, tissue maps' meaning.
READ via search excerpts only (secondary): 50/80 kVp, 512 views, 256 per kVp, fast-kV-switching, 2D breast phantom, 100-case test set, 18 teams, RMSE metric.
UNVERIFIED: spectra, energy bins, detector model, geometry distances/pitch, noise model, units of transmission and tissue maps (fractions assumed), test-set truth release, exact metric definition, whether starting_kit contains runnable code, md5 correctness of the downloaded bytes (not downloaded).

## Relevance to a CT-synth platform's data contracts
Could ingest directly:
- Ground-truth material maps (3 channels, 512x512 float32, fractions): maps onto a "material/phantom volume" contract (2D slices, 1000 samples).
- Dual-kVp line-integral/transmission sinograms 256x1024 per kVp (interlaced views, needs a view-angle index: even/odd of 512 over 360 deg, assumed).
- Baseline FBP images per kVp for image-domain tasks and sanity checks.
- Provenance: DOI, CC-BY-4.0 (allows redistribution/derivatives with attribution; cite Sidky & Pan 2024 and the Zenodo DOI).
Missing or needing work:
- Spectra / energy-bin definitions and a detector model are only in starting_kit (unread); a synth platform needs them to re-simulate or validate; must open starting_kit to confirm.
- 2D only, one anatomy (breast), 2 kVp, 3 materials; no 3D, no photon-counting multi-bin, no scatter/noise/ring-artifact variants stated.
- No explicit metadata sidecar (geometry, units, view angles) in the npy files; the contract must supply them from the kit.
- Test set and challenge evaluation code are not in this record; metrics would need reimplementation (RMSE per tissue is simple).
- Size: ~3.4 GB gz, ~ >10 GB unzipped (1000x512x512x4 B = 1.05 GB per image array; 1000x256x1024x4 = 1.05 GB per transmission array).

Suggested next step (not done): download starting_kit.tgz (103 MB, needs approval above 50 MB limit) and verify md5 7aaf6b55..., then extract spectra/geometry into a spec note.
