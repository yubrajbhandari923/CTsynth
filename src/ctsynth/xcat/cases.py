"""Locate COPD XCAT phantoms and their material files from the TrueCT folder layout.

Layout: <root>/phantom_<id>_<a>_<b>_<c>_8bit_textured_copd_eroded_flipped.bin (uint8, array shape
(c, b, a) = (z, y, x)),
<root>/material_tables/challenge_materials_copd_smeared_<id>.xlsx,
<root>/material_lac/<id>/{conversion_table_*.in, lin_att_*.bin}.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_BIN = re.compile(r"^phantom_(\d+)_(\d+)_(\d+)_(\d+)_8bit_textured_copd_eroded_flipped\.bin$")


@dataclass(frozen=True)
class CopdCase:
    patient_id: str
    bin_path: Path
    shape_zyx: tuple[int, int, int]
    xlsx: Path
    lac_bin: Path
    conversion_in: Path


def list_copd_cases(root: str | Path, require_all_files: bool = True) -> list[CopdCase]:
    """All complete cases under `root`, sorted by patient ID (incomplete ones are skipped unless
    require_all_files=False)."""
    root = Path(root)
    out = []
    for p in sorted(root.glob("phantom_*_copd_eroded_flipped.bin")):
        m = _BIN.match(p.name)
        if not m:
            continue
        pid, a, b, c = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        case = CopdCase(
            pid,
            p,
            (c, b, a),
            root / "material_tables" / f"challenge_materials_copd_smeared_{pid}.xlsx",
            root / "material_lac" / pid / "lin_att_120kV_121materials.bin",
            root / "material_lac" / pid / "conversion_table_120kV_121materials.in",
        )
        if require_all_files and not all(
            f.exists() for f in (case.xlsx, case.lac_bin, case.conversion_in)
        ):
            continue
        out.append(case)
    return out


def find_copd_case(root: str | Path, patient_id: str) -> CopdCase:
    for c in list_copd_cases(root, require_all_files=False):
        if c.patient_id == str(patient_id):
            return c
    raise FileNotFoundError(f"no COPD phantom {patient_id} under {root}")
