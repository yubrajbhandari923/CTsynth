"""Content hash of a file: the one implementation used by provenance code and experiment drivers.

Array-content hashes have different semantics and live elsewhere (`paired.schema.npy_sha256`);
the config hash is `ctsynth.config.config_sha256`.
"""

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
