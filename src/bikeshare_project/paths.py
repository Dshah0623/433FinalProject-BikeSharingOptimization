from __future__ import annotations

from pathlib import Path

from .config import project_root


ROOT = project_root()
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS_FIGURES = ROOT / "results" / "figures"
RESULTS_TABLES = ROOT / "results" / "tables"
RESULTS_RECOMMENDATIONS = ROOT / "results" / "recommendations"
REPORTS = ROOT / "reports"


def ensure_project_dirs() -> None:
    for path in [
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        MODELS,
        RESULTS_FIGURES,
        RESULTS_TABLES,
        RESULTS_RECOMMENDATIONS,
        REPORTS,
    ]:
        path.mkdir(parents=True, exist_ok=True)
