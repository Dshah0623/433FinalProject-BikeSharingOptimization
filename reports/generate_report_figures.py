#!/usr/bin/env python3
"""
Generate publication-style figures for the MSE 433 final report.

Requires a completed pipeline run (run_pipeline.py) so processed data and
results tables exist.

Usage (from project root):
    python reports/generate_report_figures.py

Outputs PNG (+ optional PDF) under reports/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bikeshare_project.config import load_config
from bikeshare_project.optimization import build_availability_plan
from bikeshare_project.paths import DATA_PROCESSED, MODELS, RESULTS_RECOMMENDATIONS, RESULTS_TABLES

OUT_DIR = ROOT / "reports" / "figures"
DPI = 300

# Report-friendly style
STYLE = {
    "figure.figsize": (7.5, 4.5),
    "figure.dpi": DPI,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
}


def _setup():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    sns.set_theme(style="whitegrid", palette="deep")


def fig1_hourly_profile(clean: pd.DataFrame) -> None:
    by_hr = clean.groupby("hr", as_index=False)["cnt"].mean()
    fig, ax = plt.subplots()
    ax.plot(by_hr["hr"], by_hr["cnt"], color="#4f46e5", linewidth=2.2, marker="o", markersize=4)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average hourly rentals (system total)")
    ax.set_title("Figure 1 — Hourly demand profile (average by hour)")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_01_hourly_profile.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2_weather(clean: pd.DataFrame) -> None:
    w = clean.groupby("weathersit", as_index=False)["cnt"].mean().sort_values("weathersit")
    fig, ax = plt.subplots()
    ax.bar(w["weathersit"].astype(str), w["cnt"], color="#0369a1", edgecolor="#0f172a", linewidth=0.4)
    ax.set_xlabel("Weather situation code (dataset)")
    ax.set_ylabel("Average hourly rentals")
    ax.set_title("Figure 2 — Average demand by weather condition")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_02_weather.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig3_temp_scatter(clean: pd.DataFrame) -> None:
    """Smoothed mean demand vs temperature (binned) + jitter sample for density."""
    rng = np.random.default_rng(42)
    sample = clean.sample(min(8000, len(clean)), random_state=42)
    # Bin temp (normalized 0–1) into 20 bins, mean cnt per bin
    clean = clean.copy()
    clean["temp_bin"] = pd.cut(clean["temp"], bins=24, include_lowest=True)
    curve = clean.groupby("temp_bin", observed=True)["cnt"].agg(["mean", "count"]).reset_index()
    curve["temp_mid"] = curve["temp_bin"].apply(lambda b: b.mid if pd.notna(b) else np.nan)

    fig, ax = plt.subplots()
    ax.scatter(
        sample["temp"],
        sample["cnt"],
        alpha=0.12,
        s=8,
        color="#94a3b8",
        label="Hourly observations (sample)",
    )
    ax.plot(curve["temp_mid"], curve["mean"], color="#b45309", linewidth=2.5, label="Mean in temp bin")
    ax.set_xlabel("Normalized temperature")
    ax.set_ylabel("Hourly rentals")
    ax.set_title("Figure 3 — Demand vs temperature (nonlinear pattern)")
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_03_temp_vs_demand.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig4_model_rmse() -> None:
    hm = pd.read_csv(RESULTS_TABLES / "holdout_metrics.csv")
    overall = hm.loc[(hm["segment_type"] == "overall") & (hm["segment_value"] == "all")].copy()
    order = ["seasonal_naive", "linear_ridge", "poisson", "boosted_tree"]
    overall["model"] = pd.Categorical(overall["model"], categories=order, ordered=True)
    overall = overall.sort_values("model")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#64748b", "#94a3b8", "#6366f1", "#4f46e5"]
    bars = ax.bar(overall["model"].astype(str), overall["rmse"], color=colors, edgecolor="#0f172a", linewidth=0.5)
    bars[3].set_color("#312e81")  # emphasize winner
    ax.set_ylabel("Holdout RMSE")
    ax.set_xlabel("Model")
    ax.set_title("Figure 4 — Model performance comparison (holdout RMSE)")
    for i, (_, row) in enumerate(overall.iterrows()):
        ax.text(i, row["rmse"] + 3, f"{row['rmse']:.1f}", ha="center", fontsize=9, fontweight="600")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_04_model_rmse.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig5_pred_vs_actual() -> None:
    with (MODELS / "best_model_metadata.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    best = meta["best_model_name"]
    hp = pd.read_csv(RESULTS_TABLES / "holdout_predictions.csv", parse_dates=["timestamp", "date"])
    df = hp.loc[hp["model"] == best].copy()
    # Subsample if huge for plotting
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    mx = max(df["cnt"].max(), df["prediction"].max())
    ax.scatter(df["cnt"], df["prediction"], alpha=0.25, s=12, color="#4f46e5", edgecolors="none")
    ax.plot([0, mx], [0, mx], "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Actual hourly rentals")
    ax.set_ylabel("Predicted hourly rentals")
    ax.set_title(f"Figure 5 — Predicted vs actual (holdout, {best})")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_05_pred_vs_actual.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig6_feature_importance() -> None:
    fi = pd.read_csv(RESULTS_TABLES / "feature_importance.csv")
    fi = fi.loc[fi["model"] == "boosted_tree"].copy()
    if fi.empty:
        print("Warning: No boosted_tree feature importance; skipping Figure 6.")
        return
    top = fi.sort_values("importance_mean", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(top))
    ax.barh(y, top["importance_mean"], color="#047857", height=0.7, xerr=top["importance_std"], capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (↓ RMSE)")
    ax.set_title("Figure 6 — Feature importance (boosted tree, holdout-based)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_06_feature_importance.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig7_allocation_vs_demand(config: dict) -> None:
    plan = pd.read_csv(RESULTS_RECOMMENDATIONS / "availability_plan.csv", parse_dates=["timestamp", "date"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(plan["hr"], plan["prediction"], label="Predicted demand", color="#0369a1", linewidth=2)
    ax.plot(plan["hr"], plan["planned_availability"], label="Optimized allocation", color="#b45309", linewidth=2)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Bikes")
    ax.set_title("Figure 7 — Predicted demand vs optimized availability (reference day)")
    ax.legend(loc="best")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_07_allocation_vs_demand.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig8_demand_multiplier_vs_bikes(config: dict) -> None:
    """Sweep multipliers on pipeline forecast; total planned bikes vs multiplier."""
    forecast_df = pd.read_csv(DATA_PROCESSED / "next_day_forecasts.csv", parse_dates=["timestamp", "date"])
    cv = pd.read_csv(DATA_PROCESSED / "cv_predictions.csv", parse_dates=["timestamp", "date"])
    with (MODELS / "best_model_metadata.json").open("r", encoding="utf-8") as f:
        best = json.load(f)["best_model_name"]
    best_cv = cv.loc[cv["model"] == best].copy()
    residuals = best_cv["cnt"] - best_cv["prediction"]

    opt = config["optimization"]
    shortage = float(opt["shortage_cost"])
    idle = float(opt["idle_cost"])
    fleet = int(opt["total_fleet_size"])

    multipliers = np.linspace(0.85, 1.25, 17)
    totals = []
    for m in multipliers:
        fd = forecast_df.copy()
        fd["prediction"] = fd["prediction"] * m
        plan = build_availability_plan(fd, residuals, shortage, idle, fleet)
        totals.append(plan["planned_availability"].sum())

    fig, ax = plt.subplots()
    ax.plot(multipliers, totals, color="#7c3aed", linewidth=2.2, marker="o", markersize=4)
    ax.set_xlabel("Demand multiplier (applied to hourly forecast)")
    ax.set_ylabel("Total planned availability (sum over 24h)")
    ax.set_title("Figure 8 — Total optimized allocation vs demand scale")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_08_demand_scale_vs_bikes.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig9_shortage_cost_sensitivity(config: dict) -> None:
    """Shortage cost sweep: mean hourly planned availability vs shortage cost."""
    forecast_df = pd.read_csv(DATA_PROCESSED / "next_day_forecasts.csv", parse_dates=["timestamp", "date"])
    cv = pd.read_csv(DATA_PROCESSED / "cv_predictions.csv", parse_dates=["timestamp", "date"])
    with (MODELS / "best_model_metadata.json").open("r", encoding="utf-8") as f:
        best = json.load(f)["best_model_name"]
    best_cv = cv.loc[cv["model"] == best].copy()
    residuals = best_cv["cnt"] - best_cv["prediction"]

    opt = config["optimization"]
    idle = float(opt["idle_cost"])
    fleet = int(opt["total_fleet_size"])

    shortage_grid = np.linspace(1.0, 12.0, 23)
    means = []
    for sc in shortage_grid:
        plan = build_availability_plan(forecast_df, residuals, float(sc), idle, fleet)
        means.append(plan["planned_availability"].mean())

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(shortage_grid, means, color="#c2410c", linewidth=2.5, marker="o", markersize=3)
    ax.axvline(float(opt["shortage_cost"]), color="#94a3b8", linestyle="--", label=f"Config default ({opt['shortage_cost']})")
    ax.set_xlabel("Shortage cost weight (vs idle = 1)")
    ax.set_ylabel("Mean hourly planned availability")
    ax.set_title("Figure 9 — Sensitivity of allocation to shortage cost assumption")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_09_shortage_cost_sensitivity.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    _setup()
    config = load_config()

    clean_path = DATA_PROCESSED / "clean_hourly.csv"
    if not clean_path.exists():
        raise SystemExit(
            f"Missing {clean_path}. Run `python run_pipeline.py` from the project root first."
        )

    clean = pd.read_csv(clean_path, parse_dates=["dteday", "timestamp", "date"])
    print("Generating Figure 1…")
    fig1_hourly_profile(clean)
    print("Generating Figure 2…")
    fig2_weather(clean)
    print("Generating Figure 3…")
    fig3_temp_scatter(clean)
    print("Generating Figure 4…")
    fig4_model_rmse()
    print("Generating Figure 5…")
    fig5_pred_vs_actual()
    print("Generating Figure 6…")
    fig6_feature_importance()
    print("Generating Figure 7…")
    fig7_allocation_vs_demand(config)
    print("Generating Figure 8…")
    fig8_demand_multiplier_vs_bikes(config)
    print("Generating Figure 9…")
    fig9_shortage_cost_sensitivity(config)

    print(f"\nDone. Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
