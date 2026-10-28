"""Print private DICOM-CT-PD tags (raw bytes + float/int guesses) for a few views of a projection series."""
import struct
import sys
from pathlib import Path

import pydicom

series = Path(sys.argv[1])
views = [int(v) for v in sys.argv[2:]] or [1, 2]
for v in views:
    d = pydicom.dcmread(series / f"proj_{v:04d}.dcm")
    print(f"--- view {v}  instance={d.get('InstanceNumber')} slope={d.RescaleSlope} icpt={d.RescaleIntercept}")
    for e in d:
        if e.tag.group < 0x7000 or e.tag.group & 1 == 0:
            continue
        b = e.value if isinstance(e.value, bytes) else bytes(str(e.value), "ascii")
        guess = ""
        if len(b) == 4:
            guess = f"f32={struct.unpack('<f', b)[0]:.6g} u32={struct.unpack('<I', b)[0]}"
        elif len(b) == 2:
            guess = f"u16={struct.unpack('<H', b)[0]}"
        elif len(b) == 8:
            guess = f"f64={struct.unpack('<d', b)[0]:.6g}"
        print(f"{e.tag} {b[:24]!r:34} {guess}")
