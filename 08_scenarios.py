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
from bikeshare_project.optimization import build_availability_plan, derive_zone_targets, solve_zone_rebalancing
from bikeshare_project.paths import DATA_PROCESSED, MODELS, RESULTS_RECOMMENDATIONS, RESULTS_TABLES


def _apply_weather_scenario(df: pd.DataFrame, scenario_cfg: dict) -> pd.DataFrame:
    adjusted = df.copy()
    adjusted["temp"] = (adjusted["temp"] + scenario_cfg.get("temp_delta", 0)).clip(lower=0, upper=1)
    adjusted["hum"] = (adjusted["hum"] + scenario_cfg.get("hum_delta", 0)).clip(lower=0, upper=1)
    adjusted["windspeed"] = (adjusted["windspeed"] + scenario_cfg.get("windspeed_delta", 0)).clip(lower=0)
    adjusted["weathersit"] = (adjusted["weathersit"] + scenario_cfg.get("weathersit_shift", 0)).clip(upper=3)
    adjusted["temp_workingday"] = adjusted["temp"] * adjusted["workingday"]
    adjusted["hum_weathersit"] = adjusted["hum"] * adjusted["weathersit"]
    return adjusted


def main() -> None:
    config = load_config()
    with (MODELS / "best_model_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    best_model_name = metadata["best_model_name"]

    estimator = None if best_model_name == "seasonal_naive" else load_joblib(MODELS / "best_model.joblib")
    forecast_df = pd.read_csv(DATA_PROCESSED / "next_day_forecasts.csv", parse_dates=["timestamp", "date"])
    cv_predictions = pd.read_csv(DATA_PROCESSED / "cv_predictions.csv", parse_dates=["timestamp", "date"])
    best_cv = cv_predictions.loc[cv_predictions["model"] == best_model_name].copy()
    residuals = best_cv["cnt"] - best_cv["prediction"]
    scenario_rows = []

    for scenario_name, scenario_cfg in config["scenarios"].items():
        scenario_features = forecast_df.copy()
        if scenario_name == "bad_weather":
            scenario_features = _apply_weather_scenario(scenario_features, scenario_cfg)

        if best_model_name == "seasonal_naive":
            scenario_features["prediction"] = np.where(
                scenario_features["lag_168"].notna(),
                scenario_features["lag_168"],
                scenario_features["lag_24"],
            )
        else:
            preds = estimator.predict(scenario_features[get_feature_columns(config)])
            if best_model_name == "linear_ridge":
                preds = np.expm1(preds)
            scenario_features["prediction"] = np.clip(preds, a_min=0, a_max=None)

        if "demand_multiplier" in scenario_cfg:
            scenario_features["prediction"] *= scenario_cfg["demand_multiplier"]
        if "forecast_multiplier" in scenario_cfg:
            scenario_features["prediction"] *= scenario_cfg["forecast_multiplier"]

        shortage_cost = float(scenario_cfg.get("shortage_cost", config["optimization"]["shortage_cost"]))
        idle_cost = float(scenario_cfg.get("idle_cost", config["optimization"]["idle_cost"]))
        fleet_size = float(scenario_cfg.get("total_fleet_size", config["optimization"]["total_fleet_size"]))
        move_cost = float(scenario_cfg.get("move_cost", config["optimization"]["move_cost"]))

        availability_plan = build_availability_plan(
            forecast_df=scenario_features,
            residuals=residuals,
            shortage_cost=shortage_cost,
            idle_cost=idle_cost,
            fleet_size=fleet_size,
        )
        zone_targets = derive_zone_targets(availability_plan, config)
        zone_summary, move_plan = solve_zone_rebalancing(
            zone_targets=zone_targets,
            current_inventory={zone: int(value) for zone, value in config["optimization"]["default_current_inventory"].items()},
            move_cost=move_cost,
            shortage_cost=shortage_cost,
            idle_cost=idle_cost,
            move_capacity=int(config["optimization"]["move_capacity"]),
        )
        scenario_rows.append(
            {
                "scenario": scenario_name,
                "forecast_total": availability_plan["prediction"].sum(),
                "planned_availability_total": availability_plan["planned_availability"].sum(),
                "proxy_shortage_total": availability_plan.get("proxy_shortage", pd.Series(dtype=float)).sum(),
                "proxy_idle_total": availability_plan.get("proxy_idle", pd.Series(dtype=float)).sum(),
                "weighted_cost_index": availability_plan.get("weighted_cost_index", pd.Series(dtype=float)).sum(),
                "service_level_proxy_mean": availability_plan.get("service_level_proxy", pd.Series(dtype=float)).mean(),
                "utilization_proxy_mean": availability_plan.get("utilization_proxy", pd.Series(dtype=float)).mean(),
                "bikes_moved": move_plan["bikes_moved"].sum(),
                "zone_shortage_slack": zone_summary["shortage_slack"].sum(),
                "zone_excess_slack": zone_summary["excess_slack"].sum(),
            }
        )

    scenario_results = pd.DataFrame(scenario_rows).sort_values("weighted_cost_index")
    save_dataframe(scenario_results, RESULTS_TABLES / "scenario_results.csv")
    print("Saved scenario comparison results.")


if __name__ == "__main__":
    main()
