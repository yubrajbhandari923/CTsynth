"""Readable summary of an imported run."""

from pathlib import Path


def build_report(run_id: str, runs_root: Path) -> str:
    """Markdown for `<runs_root>/<run_id>/`: experiment, FULL git commit (the commit the run was
    made from, not the repo HEAD), config_sha256, status (FAILED stated loudly with exit code),
    started/ended, slurm job, a metrics table (numeric leaves of metrics.json, nested keys
    dotted, sorted), figure paths (*.png/*.pdf/*.svg), checkpoint refs (path, bytes, sha256,
    external flag, location). Raises FileNotFoundError for an unknown run id."""
    raise NotImplementedError


def write_report(run_id: str, runs_root: Path) -> Path:
    """Write build_report to `<runs_root>/<run_id>.report.md` (outside the run dir, which stays
    byte-identical to what was verified). Overwriting an older report of the same run is fine."""
    raise NotImplementedError
