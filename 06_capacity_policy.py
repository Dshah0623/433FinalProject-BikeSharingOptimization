from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bikeshare_project.config import load_config
from bikeshare_project.io_utils import save_dataframe
from bikeshare_project.optimization import build_availability_plan, summarize_plan_vs_baseline
from bikeshare_project.paths import DATA_PROCESSED, RESULTS_RECOMMENDATIONS


def main() -> None:
    config = load_config()
    forecast_df = pd.read_csv(DATA_PROCESSED / "next_day_forecasts.csv", parse_dates=["timestamp", "date"])
    cv_predictions = pd.read_csv(DATA_PROCESSED / "cv_predictions.csv", parse_dates=["timestamp", "date"])
    baseline_reference = pd.read_parquet(DATA_PROCESSED / "model_table.parquet")
    with (Path(__file__).resolve().parent / "models" / "best_model_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    best_cv = cv_predictions.loc[cv_predictions["model"] == metadata["best_model_name"]].copy()
    residuals = best_cv["cnt"] - best_cv["prediction"]
    shortage_cost = float(config["optimization"]["shortage_cost"])
    idle_cost = float(config["optimization"]["idle_cost"])
    fleet_size = float(config["optimization"]["total_fleet_size"])

    availability_plan = build_availability_plan(
        forecast_df=forecast_df,
        residuals=residuals,
        shortage_cost=shortage_cost,
        idle_cost=idle_cost,
        fleet_size=fleet_size,
    )

    hourly_baseline = baseline_reference.loc[baseline_reference["date"] < pd.Timestamp(config["modeling"]["holdout_start_date"])].groupby("hr")["cnt"].mean()
    baseline_plan = forecast_df.copy()
    baseline_plan["prediction"] = baseline_plan["hr"].map(hourly_baseline)
    baseline_plan = build_availability_plan(
        forecast_df=baseline_plan,
        residuals=residuals,
        shortage_cost=shortage_cost,
        idle_cost=idle_cost,
        fleet_size=fleet_size,
    )

    policy_summary = summarize_plan_vs_baseline(availability_plan, baseline_plan, shortage_cost, idle_cost)
    save_dataframe(availability_plan, RESULTS_RECOMMENDATIONS / "availability_plan.csv")
    save_dataframe(policy_summary, RESULTS_RECOMMENDATIONS / "policy_cost_breakdown.csv")
    print("Saved forecast-driven availability plan and baseline comparison.")


if __name__ == "__main__":
    main()
