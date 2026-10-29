"""Build data/element_mass_atten_xraylib.json: total mass attenuation (cm^2/g, incl. coherent scattering) per element on a 0.5 keV grid.
Run on the laptop: uv run --no-project --with xraylib --with numpy python build_element_table.py
The DRCC jobs only read the JSON (no xraylib there). Elements = the union of the columns of all TrueCT material xlsx files.
Validated elsewhere against DukeSim's own per-organ lin_att_120kV_121materials.bin at 70 keV (agreement to 3 decimals, COPD 22728)."""
import json
from pathlib import Path

import numpy as np
import xraylib

Z = [1, 6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 26, 53]
E = np.round(np.arange(5.0, 150.0 + 1e-9, 0.5), 3)
out = {"quantity": "xraylib.CS_Total(Z,E): total mass attenuation coefficient, cm^2/g, incl. Rayleigh", "xraylib_version": "4.3.0",
       "energies_keV": E.tolist(), "elements": {str(z): [xraylib.CS_Total(z, float(e)) for e in E] for z in Z},
       "water_H2O_CS_Total_CP": [xraylib.CS_Total_CP("H2O", float(e)) for e in E],
       "caveats": "xraylib data lineage is not stated by the package (see data/tables/provenance_v1.json); interpolate log-log, avoid K edges (I: 33.17 keV)"}
Path(__file__).with_name("data").mkdir(exist_ok=True)
Path(__file__).with_name("data").joinpath("element_mass_atten_xraylib.json").write_text(json.dumps(out))
print(len(E), "energies;", "kappa_H2O(70 keV) =", out["water_H2O_CS_Total_CP"][int(np.argmin(abs(E - 70)))])
