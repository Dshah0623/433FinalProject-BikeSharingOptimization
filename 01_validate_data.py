from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bikeshare_project.config import load_config
from bikeshare_project.data import build_validation_artifacts, load_daily_data, load_hourly_data
from bikeshare_project.io_utils import copy_raw_inputs, save_dataframe
from bikeshare_project.paths import DATA_INTERIM, DATA_PROCESSED, ensure_project_dirs


def main() -> None:
    config = load_config()
    ensure_project_dirs()
    copied = copy_raw_inputs(config)
    hourly_df = load_hourly_data(str(copied["hour"]))
    daily_df = load_daily_data(str(copied["day"]))
    artifacts = build_validation_artifacts(hourly_df, daily_df, config)

    save_dataframe(artifacts.validation_summary, DATA_INTERIM / "validation_summary.csv")
    save_dataframe(artifacts.partial_day_log, DATA_INTERIM / "partial_day_log.csv")
    save_dataframe(artifacts.anomaly_log, DATA_INTERIM / "anomaly_log.csv")
    save_dataframe(artifacts.clean_hourly, DATA_PROCESSED / "clean_hourly.csv")
    print("Saved validation outputs to data/interim and data/processed.")


if __name__ == "__main__":
    main()
