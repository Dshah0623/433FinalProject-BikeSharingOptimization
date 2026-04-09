from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bikeshare_project.config import load_config
from bikeshare_project.features import get_feature_columns
from bikeshare_project.io_utils import load_joblib, save_dataframe
from bikeshare_project.paths import DATA_PROCESSED, MODELS


def main() -> None:
    config = load_config()
    model_df = pd.read_parquet(DATA_PROCESSED / "model_table.parquet")
    with (MODELS / "best_model_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    best_model_name = metadata["best_model_name"]
    reference_date = pd.Timestamp(config["modeling"]["forecast_reference_date"])
    available_dates = sorted(model_df["date"].drop_duplicates())
    if reference_date not in set(available_dates):
        reference_date = available_dates[-1]

    if best_model_name == "seasonal_naive":
        reference_df = model_df.loc[model_df["date"] == reference_date].copy()
        reference_df["prediction"] = np.where(reference_df["lag_168"].notna(), reference_df["lag_168"], reference_df["lag_24"])
    else:
        estimator = load_joblib(MODELS / "best_model.joblib")
        reference_df = model_df.loc[model_df["date"] == reference_date].copy()
        if reference_df.empty:
            raise ValueError(f"No complete forecast rows found for reference date {reference_date.date()}")
        preds = estimator.predict(reference_df[get_feature_columns(config)])
        if best_model_name == "linear_ridge":
            preds = np.expm1(preds)
        reference_df["prediction"] = np.clip(preds, a_min=0, a_max=None)

    cv_predictions = pd.read_csv(DATA_PROCESSED / "cv_predictions.csv", parse_dates=["timestamp", "date"])
    best_cv = cv_predictions.loc[cv_predictions["model"] == best_model_name].copy()
    best_cv["residual"] = best_cv["cnt"] - best_cv["prediction"]
    residual_quantiles = (
        best_cv["residual"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).rename_axis("quantile").reset_index(name="residual_quantile")
    )
    interval_width = abs(best_cv["residual"]).quantile(0.9)
    reference_df["prediction_lower"] = np.clip(reference_df["prediction"] - interval_width, a_min=0, a_max=None)
    reference_df["prediction_upper"] = reference_df["prediction"] + interval_width

    save_dataframe(reference_df, DATA_PROCESSED / "next_day_forecasts.csv")
    save_dataframe(residual_quantiles, DATA_PROCESSED / "forecast_intervals.csv")
    print("Saved next-day forecast and empirical uncertainty summaries.")


if __name__ == "__main__":
    main()
