"""Shared paths and case table for the TrueCT pilot. Original data are read-only; outputs go under OUT."""
from pathlib import Path

ROOT = Path("/home/yb107/datasets/truect_pilot")
OUT = ROOT / "outputs"
REF = ROOT / "reference/extracted/reference"

# NOTE liver dims are given as (nx, ny, nz) = (880, 1234, 676): the survey found its stored layout (676, 1234, 880), i.e. the
# filename lists (ny, nx, nz) for this case only.
# name -> (phantom .bin, dims (n0, n1, n2) exactly as in the filename, material xlsx, voxel mm or None if undocumented)
PHANTOMS = {
    "reference": (REF / "challenge_reference_860_860_320_0.25_mm_10mat.bin", (860, 860, 320),
                  REF / "reference_phantom_mat.xlsx", 0.25),
    "copd_22728": (ROOT / "COPD/phantom_22728_1048_1438_1084_8bit_textured_copd_eroded_flipped.bin", (1048, 1438, 1084),
                   ROOT / "COPD/material_tables/challenge_materials_copd_smeared_22728.xlsx", None),
    "lung_18460": (ROOT / "LungLesions/phantom_18460_970_1310_1240_8bit_textured_2_lung_lesions_flipped.bin", (970, 1310, 1240),
                   ROOT / "LungLesions/material_tables/challenge_materials_lung_lesion_smeared_18460.xlsx", None),
    "liver_279": (ROOT / "LiverLesions/female_pt279_1234_880_676_textured_liver_3_liver_lesions_ribs_fixed_cropped_flipped.bin",
                  (880, 1234, 676), ROOT / "LiverLesions/material_tables/challenge_materials_liver_smeared_pt279.xlsx", None),
    "copd_21535": (ROOT / "COPD/phantom_21535_872_1594_1252_8bit_textured_copd_eroded_flipped.bin", (872, 1594, 1252),
                   ROOT / "COPD/material_tables/challenge_materials_copd_smeared_21535.xlsx", None),
}
PROJECTIONS = {
    "reference": REF / "dcmproj_reference",
    "dcm_000": ROOT / "projections/dcm_000",
    "dcm_001": ROOT / "projections/dcm_001",
}
EXTRA = {  # lower priority, dims from filenames (see create_previews.py)
    "lung_18460": (ROOT / "LungLesions/phantom_18460_970_1310_1240_8bit_textured_2_lung_lesions_flipped.bin", (970, 1310, 1240)),
    "liver_279": (ROOT / "LiverLesions/female_pt279_1234_880_676_textured_liver_3_liver_lesions_ribs_fixed_cropped_flipped.bin",
                  (1234, 880, 676)),
}

# Storage layout established by survey_phantoms.py (smoothness score 0.99-1.0 vs <=0.69 for other reshapes):
# C-order [z, y, x] with the filename dims reading (nx, ny, nz).
def array_shape(dims):
    return tuple(reversed(dims))

# conversion tables (ORGAN_ID name MC_MATERIAL DENSITY) used to cross-check the xlsx
INFILES = {
    "copd_22728": ROOT / "COPD/material_lac/22728/conversion_table_120kV_121materials.in",
    "copd_21535": ROOT / "COPD/material_lac/21535/conversion_table_120kV_121materials.in",
    "lung_18460": ROOT / "LungLesions/material_lac/18460/conversion_table_120kV_121materials.in",
    "liver_279": ROOT / "LiverLesions/material_lac/pt279/conversion_table_120kV_121materials.in",
}
