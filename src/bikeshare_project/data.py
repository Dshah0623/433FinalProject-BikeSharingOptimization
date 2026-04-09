from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class ValidationArtifacts:
    clean_hourly: pd.DataFrame
    partial_day_log: pd.DataFrame
    anomaly_log: pd.DataFrame
    validation_summary: pd.DataFrame


def load_hourly_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_daily_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    return df.sort_values("dteday").reset_index(drop=True)


def prepare_hourly_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    prepared = df.copy()
    if config["modeling"]["collapse_weathersit_4_to_3"]:
        prepared["weathersit_original"] = prepared["weathersit"]
        prepared["weathersit"] = prepared["weathersit"].replace({4: 3})
    else:
        prepared["weathersit_original"] = prepared["weathersit"]
    prepared["date"] = prepared["dteday"].dt.normalize()
    prepared["year"] = prepared["dteday"].dt.year
    prepared["month_name"] = prepared["dteday"].dt.month_name().str.slice(0, 3)
    prepared["weekday_name"] = prepared["dteday"].dt.day_name().str.slice(0, 3)
    return prepared


def _build_partial_day_log(hourly_df: pd.DataFrame) -> pd.DataFrame:
    observed = hourly_df.groupby("date")["hr"].agg(lambda values: sorted(set(values))).reset_index(name="observed_hours")
    observed["observed_hour_count"] = observed["observed_hours"].apply(len)
    observed["missing_hours"] = observed["observed_hours"].apply(
        lambda hours: [hour for hour in range(24) if hour not in hours]
    )
    observed["is_complete_day"] = observed["observed_hour_count"].eq(24)
    return observed


def _build_anomaly_log(hourly_df: pd.DataFrame, partial_day_log: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    daily = (
        hourly_df.groupby("date")
        .agg(
            total_cnt=("cnt", "sum"),
            max_hourly_cnt=("cnt", "max"),
            mean_hourly_cnt=("cnt", "mean"),
        )
        .reset_index()
    )
    daily = daily.merge(partial_day_log[["date", "observed_hour_count", "is_complete_day"]], on="date", how="left")
    z = (daily["total_cnt"] - daily["total_cnt"].mean()) / daily["total_cnt"].std(ddof=0)
    daily["daily_total_zscore"] = z.fillna(0.0)
    daily["flag_extreme_daily_total"] = daily["daily_total_zscore"].abs() >= 2.5
    configured_anomalies = {pd.Timestamp(value) for value in config["modeling"]["anomaly_dates"]}
    daily["flag_configured_anomaly"] = daily["date"].isin(configured_anomalies)
    daily["is_anomaly_day"] = daily["flag_extreme_daily_total"] | daily["flag_configured_anomaly"]
    return daily.sort_values("date").reset_index(drop=True)


def build_validation_artifacts(hourly_df: pd.DataFrame, daily_df: pd.DataFrame, config: Dict[str, Any]) -> ValidationArtifacts:
    prepared = prepare_hourly_data(hourly_df, config)
    partial_day_log = _build_partial_day_log(prepared)
    anomaly_log = _build_anomaly_log(prepared, partial_day_log, config)

    prepared = prepared.merge(
        partial_day_log[["date", "observed_hour_count", "is_complete_day"]],
        on="date",
        how="left",
    )
    prepared = prepared.merge(
        anomaly_log[["date", "is_anomaly_day", "daily_total_zscore"]],
        on="date",
        how="left",
    )

    hourly_to_daily = prepared.groupby("date", as_index=False)["cnt"].sum().rename(columns={"cnt": "hourly_sum_cnt"})
    daily_compare = daily_df.rename(columns={"dteday": "date", "cnt": "daily_cnt"}).merge(hourly_to_daily, on="date", how="left")
    daily_compare["daily_match"] = np.isclose(daily_compare["daily_cnt"], daily_compare["hourly_sum_cnt"])

    summary_rows = [
        {"metric": "hourly_rows", "value": len(prepared)},
        {"metric": "daily_rows", "value": len(daily_df)},
        {"metric": "date_min", "value": prepared["date"].min().date().isoformat()},
        {"metric": "date_max", "value": prepared["date"].max().date().isoformat()},
        {"metric": "unique_dates", "value": prepared["date"].nunique()},
        {"metric": "partial_day_count", "value": int((~partial_day_log["is_complete_day"]).sum())},
        {"metric": "configured_anomaly_count", "value": int(anomaly_log["flag_configured_anomaly"].sum())},
        {"metric": "extreme_daily_total_count", "value": int(anomaly_log["flag_extreme_daily_total"].sum())},
        {"metric": "daily_aggregation_matches_day_csv", "value": bool(daily_compare["daily_match"].all())},
        {"metric": "weathersit_4_count_before_collapse", "value": int((prepared["weathersit_original"] == 4).sum())},
        {"metric": "duplicate_hourly_rows", "value": int(prepared.duplicated().sum())},
        {"metric": "missing_values_total", "value": int(prepared.isna().sum().sum())},
    ]
    validation_summary = pd.DataFrame(summary_rows)
    return ValidationArtifacts(
        clean_hourly=prepared,
        partial_day_log=partial_day_log,
        anomaly_log=anomaly_log,
        validation_summary=validation_summary,
    )
