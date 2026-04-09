from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bikeshare_project.config import load_config
from bikeshare_project.features import build_feature_dictionary, build_feature_table
from bikeshare_project.io_utils import save_dataframe
from bikeshare_project.paths import DATA_PROCESSED, RESULTS_TABLES


def main() -> None:
    config = load_config()
    df = pd.read_csv(DATA_PROCESSED / "clean_hourly.csv", parse_dates=["dteday", "timestamp", "date"])
    feature_table = build_feature_table(df, config)
    feature_dictionary = build_feature_dictionary(config)
    save_dataframe(feature_table, DATA_PROCESSED / "model_table.parquet")
    save_dataframe(feature_dictionary, RESULTS_TABLES / "feature_dictionary.csv")
    print("Saved modeling table and feature dictionary.")


if __name__ == "__main__":
    main()
