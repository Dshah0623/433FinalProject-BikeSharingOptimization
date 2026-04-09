from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bikeshare_project.config import load_config
from bikeshare_project.eda import build_eda_tables, create_eda_figures, save_eda_workbook
from bikeshare_project.paths import DATA_PROCESSED, RESULTS_FIGURES, RESULTS_TABLES


def main() -> None:
    load_config()
    df = pd.read_csv(DATA_PROCESSED / "clean_hourly.csv", parse_dates=["dteday", "timestamp", "date"])
    tables = build_eda_tables(df)
    save_eda_workbook(tables, RESULTS_TABLES / "eda_tables.xlsx")
    create_eda_figures(df, RESULTS_FIGURES)
    print("Saved EDA tables and figures.")


if __name__ == "__main__":
    main()
