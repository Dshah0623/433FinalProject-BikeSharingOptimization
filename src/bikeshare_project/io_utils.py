from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from .paths import DATA_RAW, ensure_project_dirs


def copy_raw_inputs(config: Dict[str, Any]) -> Dict[str, Path]:
    ensure_project_dirs()
    mapping = {
        "hour": (Path(config["paths"]["source_hour_csv"]), DATA_RAW / "hour.csv"),
        "day": (Path(config["paths"]["source_day_csv"]), DATA_RAW / "day.csv"),
        "readme": (Path(config["paths"]["source_readme"]), DATA_RAW / "Readme.txt"),
    }
    copied: Dict[str, Path] = {}
    for key, (source, destination) in mapping.items():
        if not source.exists():
            raise FileNotFoundError(f"Expected source file not found: {source}")
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied[key] = destination
    return copied


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def save_joblib(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index)
    elif suffix == ".parquet":
        df.to_parquet(path, index=index)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=index)
    else:
        raise ValueError(f"Unsupported dataframe output format: {path}")
