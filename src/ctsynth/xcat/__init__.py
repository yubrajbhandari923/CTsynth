"""XCAT material phantom -> attenuation -> LEAP projections -> noisy helical FBP CT, plus compact
training caches.

Needs the optional extras (torch, openpyxl) and, for the LEAP parts, LLNL LEAP (commit 0c8846f) on
a CUDA node.
Conventions: phantom arrays are uint8 [z, y, x] (array axes PROVISIONALLY S, L, P); densities
g/cm^3; mass attenuation cm^2/g;
linear attenuation cm^-1 in tables and mm^-1 for LEAP; projections (view, detector_row,
detector_channel) dimensionless line integrals.
"""
