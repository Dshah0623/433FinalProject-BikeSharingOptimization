from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "data" / "interim" / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def build_eda_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    daily_totals = df.groupby("date", as_index=False)["cnt"].sum()
    tables = {
        "summary_statistics": df[["cnt", "temp", "hum", "windspeed", "casual", "registered"]].describe().round(3),
        "hourly_profile": df.groupby("hr", as_index=False)["cnt"].mean().rename(columns={"cnt": "mean_cnt"}).round(3),
        "weekday_hour_profile": df.groupby(["weekday_name", "hr"], as_index=False)["cnt"].mean().round(3),
        "season_profile": df.groupby("season", as_index=False)["cnt"].agg(["mean", "median", "max"]).reset_index().round(3),
        "weather_profile": df.groupby("weathersit", as_index=False)["cnt"].agg(["mean", "median", "max"]).reset_index().round(3),
        "workingday_profile": df.groupby("workingday", as_index=False)["cnt"].mean().rename(columns={"cnt": "mean_cnt"}).round(3),
        "holiday_profile": df.groupby("holiday", as_index=False)["cnt"].mean().rename(columns={"cnt": "mean_cnt"}).round(3),
        "daily_totals": daily_totals.round(3),
        "casual_registered_hourly": df.groupby("hr", as_index=False)[["casual", "registered"]].mean().round(3),
    }
    return tables


def save_eda_workbook(tables: Dict[str, pd.DataFrame], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_eda_figures(df: pd.DataFrame, figures_dir: str | Path) -> None:
    figures_dir = Path(figures_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["cnt"], bins=40, kde=True, ax=ax, color="#33658A")
    ax.set_title("Hourly Bike Demand Distribution")
    ax.set_xlabel("Hourly rentals")
    _save_figure(fig, figures_dir / "01_demand_distribution.png")

    daily_totals = df.groupby("date", as_index=False)["cnt"].sum()
    anomaly_dates = df.loc[df["is_anomaly_day"] == 1, "date"].drop_duplicates()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_totals["date"], daily_totals["cnt"], color="#2F4858", linewidth=1.5)
    for anomaly_date in anomaly_dates:
        ax.axvline(anomaly_date, color="#BC4749", alpha=0.25, linestyle="--")
    ax.set_title("Daily Demand Timeline with Anomaly Markers")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total daily rentals")
    _save_figure(fig, figures_dir / "02_daily_timeline_with_anomalies.png")

    hourly = df.groupby(["workingday", "hr"], as_index=False)["cnt"].mean()
    hourly["day_type"] = hourly["workingday"].map({1: "Working day", 0: "Weekend/Holiday"})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=hourly, x="hr", y="cnt", hue="day_type", linewidth=2.5, ax=ax, palette=["#8D99AE", "#D62828"])
    ax.set_title("Hourly Demand Profile by Day Type")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean hourly rentals")
    _save_figure(fig, figures_dir / "03_hourly_profile_day_type.png")

    heatmap_data = df.pivot_table(values="cnt", index="weekday_name", columns="hr", aggfunc="mean")
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heatmap_data = heatmap_data.reindex([day for day in weekday_order if day in heatmap_data.index])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
    ax.set_title("Hourly Demand Heatmap by Weekday")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Weekday")
    _save_figure(fig, figures_dir / "04_hourly_weekday_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="season", y="cnt", hue="season", dodge=False, legend=False, ax=ax, palette="Set2")
    ax.set_title("Demand Distribution by Season")
    ax.set_xlabel("Season code")
    ax.set_ylabel("Hourly rentals")
    _save_figure(fig, figures_dir / "05_demand_by_season.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="weathersit", y="cnt", hue="weathersit", dodge=False, legend=False, ax=ax, palette="Set3")
    ax.set_title("Demand Distribution by Weather Situation")
    ax.set_xlabel("Weather situation code")
    ax.set_ylabel("Hourly rentals")
    _save_figure(fig, figures_dir / "06_demand_by_weather.png")

    mix = df.groupby("hr", as_index=False)[["casual", "registered"]].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(mix["hr"], mix["casual"], mix["registered"], labels=["Casual", "Registered"], colors=["#F4A261", "#264653"])
    ax.legend(loc="upper left")
    ax.set_title("Average Hourly Demand Mix: Casual vs Registered")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean hourly rentals")
    _save_figure(fig, figures_dir / "07_casual_registered_mix.png")
