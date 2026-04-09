from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def build_feature_table(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    modeling_df = df.copy().sort_values("timestamp").reset_index(drop=True)
    if config["modeling"]["drop_partial_days_from_modeling"]:
        modeling_df = modeling_df.loc[modeling_df["is_complete_day"] == 1].copy()

    modeling_df["hour_sin"] = np.sin(2 * np.pi * modeling_df["hr"] / 24)
    modeling_df["hour_cos"] = np.cos(2 * np.pi * modeling_df["hr"] / 24)
    modeling_df["weekday_sin"] = np.sin(2 * np.pi * modeling_df["weekday"] / 7)
    modeling_df["weekday_cos"] = np.cos(2 * np.pi * modeling_df["weekday"] / 7)
    modeling_df["month_sin"] = np.sin(2 * np.pi * modeling_df["mnth"] / 12)
    modeling_df["month_cos"] = np.cos(2 * np.pi * modeling_df["mnth"] / 12)

    modeling_df["lag_24"] = modeling_df["cnt"].shift(24)
    modeling_df["lag_168"] = modeling_df["cnt"].shift(168)
    modeling_df["rolling_mean_24"] = modeling_df["cnt"].shift(1).rolling(24, min_periods=12).mean()
    modeling_df["rolling_mean_168"] = modeling_df["cnt"].shift(1).rolling(168, min_periods=24).mean()
    modeling_df["same_hour_mean_7"] = modeling_df.groupby("hr")["cnt"].transform(
        lambda series: series.shift(1).rolling(7, min_periods=3).mean()
    )

    modeling_df["temp_workingday"] = modeling_df["temp"] * modeling_df["workingday"]
    modeling_df["hum_weathersit"] = modeling_df["hum"] * modeling_df["weathersit"]
    modeling_df["target_log1p"] = np.log1p(modeling_df["cnt"])

    feature_columns = get_feature_columns(config)
    modeling_df = modeling_df.dropna(subset=["lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168", "same_hour_mean_7"])
    modeling_df = modeling_df.reset_index(drop=True)
    modeling_df["modeling_sample"] = 1
    ordered_columns = ["timestamp", "date", "cnt", "target_log1p"] + feature_columns
    ordered_columns = list(dict.fromkeys(ordered_columns))
    return modeling_df[ordered_columns]


def get_feature_columns(config: Dict[str, Any]) -> List[str]:
    return config["features"]["categorical"] + config["features"]["numeric"]


def get_categorical_features(config: Dict[str, Any]) -> List[str]:
    return config["features"]["categorical"]


def get_numeric_features(config: Dict[str, Any]) -> List[str]:
    return config["features"]["numeric"]


def build_feature_dictionary(config: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    descriptions = {
        "season": "Season code from source dataset",
        "mnth": "Calendar month",
        "hr": "Hour of day",
        "weekday": "Day of week code",
        "workingday": "Indicator for non-weekend, non-holiday day",
        "holiday": "Holiday indicator",
        "weathersit": "Collapsed weather situation code",
        "yr": "Dataset year indicator (0=2011, 1=2012)",
        "temp": "Normalized temperature",
        "hum": "Normalized humidity",
        "windspeed": "Normalized wind speed",
        "hour_sin": "Cyclical encoding of hour",
        "hour_cos": "Cyclical encoding of hour",
        "weekday_sin": "Cyclical encoding of weekday",
        "weekday_cos": "Cyclical encoding of weekday",
        "month_sin": "Cyclical encoding of month",
        "month_cos": "Cyclical encoding of month",
        "lag_24": "Demand from same hour previous day",
        "lag_168": "Demand from same hour previous week",
        "rolling_mean_24": "Rolling average of the previous 24 observed hours",
        "rolling_mean_168": "Rolling average of the previous 168 observed hours",
        "same_hour_mean_7": "Mean of last 7 observations for the same hour of day",
        "temp_workingday": "Interaction between temperature and workingday",
        "hum_weathersit": "Interaction between humidity and weather situation",
        "is_anomaly_day": "Indicator for configured or statistical anomaly day",
        "is_complete_day": "Indicator for dates with all 24 hourly records",
    }
    for feature in get_feature_columns(config):
        rows.append(
            {
                "feature_name": feature,
                "feature_type": "categorical" if feature in get_categorical_features(config) else "numeric",
                "description": descriptions.get(feature, ""),
            }
        )
    return pd.DataFrame(rows)
