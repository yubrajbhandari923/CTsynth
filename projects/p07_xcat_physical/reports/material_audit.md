# XCAT COPD material audit (EXP-040)

67 patient tables, sha256 of (name, file hash) list `8458b3b3ce92050c...`. Compositions are identical across patients for every organ ID; only densities differ (smeared per patient).

| channel | name (first xlsx organ) | organ IDs | density range over all IDs/patients (g/cm3) | kappa 70 keV (cm2/g) | kappa 80 keV |
|---|---|---|---|---|---|
| 0 | air | [0, 70] | 0.0012-0.0012 | 0.1748 | 0.1662 |
| 1 | body | [1] | 0.9048-0.9353 | 0.1878 | 0.1805 |
| 2 | sternum | [3, 8, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59] | 0.8306-1.1926 | 0.2570 | 0.2229 |
| 3 | muscle | [4, 13, 120] | 1.0350-1.0649 | 0.1916 | 0.1823 |
| 4 | lung | [6, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 60, 106, 107, 108, 109, 110] | 0.0226-0.9520 | 0.1904 | 0.1815 |
| 5 | spine_cortical | [9, 39, 40, 41, 42, 43] | 0.9193-1.3958 | 0.2292 | 0.2056 |
| 6 | skin | [24] | 1.0744-1.1054 | 0.1908 | 0.1818 |
| 7 | aorta | [25, 68] | 1.0454-1.0747 | 0.1921 | 0.1827 |

## Attenuation-shape degeneracy: angle (deg) between kappa(E) vectors

Upper triangle: 30-140 keV (111 energies); lower triangle: only {70, 80} keV. Small angle = near-degenerate (cannot be told apart from attenuation shape; separable only through density-times-kappa magnitude AND priors).

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | - | 4.51 | 23.66 | 0.33 | 1.09 | 17.18 | 0.99 | 0.02 |
| 1 | 0.321 | - | 28.17 | 4.18 | 3.42 | 21.69 | 3.51 | 4.50 |
| 2 | 2.621 | 2.942 | - | 23.99 | 24.75 | 6.48 | 24.65 | 23.67 |
| 3 | 0.028 | 0.293 | 2.649 | - | 0.76 | 17.51 | 0.66 | 0.32 |
| 4 | 0.084 | 0.237 | 2.705 | 0.056 | - | 18.28 | 0.10 | 1.08 |
| 5 | 1.656 | 1.977 | 0.965 | 1.684 | 1.740 | - | 18.18 | 17.19 |
| 6 | 0.077 | 0.244 | 2.697 | 0.049 | 0.007 | 1.733 | - | 0.98 |
| 7 | 0.004 | 0.318 | 2.624 | 0.025 | 0.081 | 1.660 | 0.073 | - |

Singular values of the normalised kappa(E) curves, 30-140 keV: [2.78962e+00 4.66903e-01 5.46136e-04 1.97405e-04 4.22183e-06 8.82328e-07
 3.50266e-07 6.13474e-08]; at {70, 80} keV: [2.828   0.04922].
Structural (arithmetic) consequences: over 30-140 keV the attenuation shapes span about 2 dimensions (photoelectric/Compton-like), so no number of energies can identify more than ~2 independent combinations of the 8 concentrations from attenuation alone; one energy (70 keV) fixes 1 number per voxel.
