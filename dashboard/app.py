from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from theme import inject_styles, style_plotly

from bikeshare_project.config import load_config
from bikeshare_project.features import get_feature_columns
from bikeshare_project.io_utils import load_joblib
from bikeshare_project.optimization import build_availability_plan, derive_zone_targets, solve_zone_rebalancing
from bikeshare_project.paths import DATA_PROCESSED, MODELS, RESULTS_RECOMMENDATIONS, RESULTS_TABLES


st.set_page_config(
    page_title="Bike-share · MSE 433",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()


@st.cache_data
def load_artifacts() -> dict:
    config = load_config()
    clean_hourly = pd.read_csv(DATA_PROCESSED / "clean_hourly.csv", parse_dates=["dteday", "timestamp", "date"])
    model_table = pd.read_parquet(DATA_PROCESSED / "model_table.parquet")
    next_day_forecasts = pd.read_csv(DATA_PROCESSED / "next_day_forecasts.csv", parse_dates=["timestamp", "date"])
    forecast_intervals = pd.read_csv(DATA_PROCESSED / "forecast_intervals.csv")
    availability_plan = pd.read_csv(RESULTS_RECOMMENDATIONS / "availability_plan.csv", parse_dates=["timestamp", "date"])
    policy_cost_breakdown = pd.read_csv(RESULTS_RECOMMENDATIONS / "policy_cost_breakdown.csv")
    zone_targets = pd.read_csv(RESULTS_RECOMMENDATIONS / "zone_targets.csv")
    zone_move_plan = pd.read_csv(RESULTS_RECOMMENDATIONS / "zone_move_plan.csv")
    zone_plan_summary = pd.read_csv(RESULTS_RECOMMENDATIONS / "zone_plan_summary.csv")
    zone_sensitivity = pd.read_csv(RESULTS_RECOMMENDATIONS / "zone_sensitivity.csv")
    scenario_results = pd.read_csv(RESULTS_TABLES / "scenario_results.csv")
    cv_predictions = pd.read_csv(DATA_PROCESSED / "cv_predictions.csv", parse_dates=["timestamp", "date"])
    holdout_metrics = pd.read_csv(RESULTS_TABLES / "holdout_metrics.csv")
    feature_importance = pd.read_csv(RESULTS_TABLES / "feature_importance.csv")
    with (MODELS / "best_model_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "config": config,
        "clean_hourly": clean_hourly,
        "model_table": model_table,
        "next_day_forecasts": next_day_forecasts,
        "forecast_intervals": forecast_intervals,
        "availability_plan": availability_plan,
        "policy_cost_breakdown": policy_cost_breakdown,
        "zone_targets": zone_targets,
        "zone_move_plan": zone_move_plan,
        "zone_plan_summary": zone_plan_summary,
        "zone_sensitivity": zone_sensitivity,
        "scenario_results": scenario_results,
        "cv_predictions": cv_predictions,
        "holdout_metrics": holdout_metrics,
        "feature_importance": feature_importance,
        "metadata": metadata,
    }


@st.cache_resource
def load_best_model():
    metadata_path = MODELS / "best_model_metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["best_model_name"] == "seasonal_naive":
        return None
    return load_joblib(MODELS / "best_model.joblib")


def recalculate_live_plan(artifacts: dict, shortage_cost: float, idle_cost: float, fleet_size: int) -> pd.DataFrame:
    best_model_name = artifacts["metadata"]["best_model_name"]
    best_cv = artifacts["cv_predictions"].loc[artifacts["cv_predictions"]["model"] == best_model_name].copy()
    residuals = best_cv["cnt"] - best_cv["prediction"]
    live_plan = build_availability_plan(
        forecast_df=artifacts["next_day_forecasts"],
        residuals=residuals,
        shortage_cost=shortage_cost,
        idle_cost=idle_cost,
        fleet_size=fleet_size,
    )
    return live_plan


artifacts = load_artifacts()
config = artifacts["config"]
best_model = load_best_model()

st.markdown(
    f"""
<div class="velo-hero">
  <div class="velo-hero-inner">
    <div class="velo-kicker">MSE 433</div>
    <h1 class="velo-title">Demand, availability, and zone rebalancing</h1>
    <p class="velo-sub">Next-day hourly forecasts, fleet availability targets, overnight zone moves, and scenario results from the project pipeline.</p>
    <div class="velo-meta">
      <span class="velo-pill">Best model · {artifacts["metadata"]["best_model_name"]}</span>
      <span class="velo-pill">Stages · validation through scenarios</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<p class="velo-sidebar-brand">Parameters</p><p class="velo-sidebar-title">Costs and zones</p><div class="velo-sidebar-rule"></div>',
        unsafe_allow_html=True,
    )
    shortage_cost = st.slider("Shortage penalty", min_value=1.0, max_value=12.0, value=float(config["optimization"]["shortage_cost"]), step=0.5)
    idle_cost = st.slider("Idle cost", min_value=0.5, max_value=5.0, value=float(config["optimization"]["idle_cost"]), step=0.5)
    move_cost = st.slider("Move cost", min_value=0.1, max_value=3.0, value=float(config["optimization"]["move_cost"]), step=0.1)
    fleet_size = st.slider("Available fleet size", min_value=500, max_value=1300, value=int(config["optimization"]["total_fleet_size"]), step=25)
    include_anomaly_day = st.toggle("Include anomaly day in interpretation", value=False)
    current_inventory = {
        "Residential": st.number_input("Residential current inventory", min_value=0, value=int(config["optimization"]["default_current_inventory"]["Residential"])),
        "Employment Core": st.number_input("Employment Core current inventory", min_value=0, value=int(config["optimization"]["default_current_inventory"]["Employment Core"])),
        "Leisure/Mixed Use": st.number_input("Leisure/Mixed Use current inventory", min_value=0, value=int(config["optimization"]["default_current_inventory"]["Leisure/Mixed Use"])),
    }
    st.markdown('<p style="color:#94a3b8;font-size:0.82rem;margin:1rem 0 0.5rem 0;font-weight:600;">Zone share assumptions</p>', unsafe_allow_html=True)
    residential_share = st.slider("Residential share", 0.0, 1.0, float(config["optimization"]["default_zone_shares"]["workingday"]["Residential"]), 0.01)
    employment_share = st.slider("Employment Core share", 0.0, 1.0, float(config["optimization"]["default_zone_shares"]["workingday"]["Employment Core"]), 0.01)
    leisure_share = max(0.0, round(1.0 - residential_share - employment_share, 2))
    st.write(f"Leisure/Mixed Use share: `{leisure_share:.2f}`")
    zone_shares = {
        "Residential": residential_share,
        "Employment Core": employment_share,
        "Leisure/Mixed Use": leisure_share,
    }

live_plan = recalculate_live_plan(artifacts, shortage_cost, idle_cost, fleet_size)
zone_targets = derive_zone_targets(live_plan, config, zone_shares=zone_shares)
zone_summary, zone_move_plan = solve_zone_rebalancing(
    zone_targets=zone_targets,
    current_inventory=current_inventory,
    move_cost=move_cost,
    shortage_cost=shortage_cost,
    idle_cost=idle_cost,
    move_capacity=int(config["optimization"]["move_capacity"]),
)

tabs = st.tabs(
    [
        "Historical Demand Insights",
        "Next-Day Demand Forecast",
        "Allocation / Rebalancing",
        "Scenario Testing",
        "Model Diagnostics",
        "Business Impact",
    ]
)

with tabs[0]:
    st.markdown(
        '<p class="velo-section-title">Historical demand</p><p class="velo-section-hint">Spot trends, intraday shape, and weather sensitivity. Toggle anomaly days from the sidebar when comparing to “normal” operations.</p>',
        unsafe_allow_html=True,
    )
    clean_hourly = artifacts["clean_hourly"].copy()
    if not include_anomaly_day:
        clean_hourly = clean_hourly.loc[clean_hourly["is_anomaly_day"] == 0]
    daily_totals = clean_hourly.groupby("date", as_index=False)["cnt"].sum()
    fig = px.line(daily_totals, x="date", y="cnt", title="Daily total demand")
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    hourly_profile = clean_hourly.groupby(["workingday", "hr"], as_index=False)["cnt"].mean()
    hourly_profile["day_type"] = hourly_profile["workingday"].map({1: "Working day", 0: "Weekend/Holiday"})
    fig = px.line(hourly_profile, x="hr", y="cnt", color="day_type", title="Hourly profile by day type")
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    weather_profile = clean_hourly.groupby("weathersit", as_index=False)["cnt"].mean()
    fig = px.bar(weather_profile, x="weathersit", y="cnt", title="Average demand by weather code")
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.markdown(
        '<p class="velo-section-title">Next-day forecast</p><p class="velo-section-hint">Reference run from the pipeline, with empirical uncertainty band and optional actuals when present.</p>',
        unsafe_allow_html=True,
    )
    forecast_df = artifacts["next_day_forecasts"].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["prediction"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#2dd4bf", width=2),
            marker=dict(size=4, color="#2dd4bf"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(forecast_df["timestamp"]) + list(forecast_df["timestamp"][::-1]),
            y=list(forecast_df["prediction_upper"]) + list(forecast_df["prediction_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(45, 212, 191, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Empirical interval",
        )
    )
    if "cnt" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df["cnt"],
                mode="lines",
                name="Actual",
                line=dict(dash="dash", color="#a78bfa", width=1.5),
            )
        )
    fig.update_layout(title="Reference next-day forecast")
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Forecast total", f"{forecast_df['prediction'].sum():,.0f}")
    kpi_cols[1].metric("Peak-hour forecast", f"{forecast_df['prediction'].max():,.0f}")
    kpi_cols[2].metric("Mean forecast", f"{forecast_df['prediction'].mean():.1f}")
    kpi_cols[3].metric("Best model", artifacts["metadata"]["best_model_name"])

with tabs[2]:
    st.markdown(
        '<p class="velo-section-title">Allocation & rebalancing</p><p class="velo-section-hint">Live plan from sidebar costs and fleet size; heatmap shows overnight flows between zones.</p>',
        unsafe_allow_html=True,
    )
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Planned availability total", f"{live_plan['planned_availability'].sum():,.0f}")
    kpi_cols[1].metric("Proxy shortage", f"{live_plan['proxy_shortage'].sum():,.0f}")
    kpi_cols[2].metric("Proxy idle", f"{live_plan['proxy_idle'].sum():,.0f}")
    kpi_cols[3].metric("Bikes moved", f"{zone_move_plan['bikes_moved'].sum():,.0f}")

    fig = px.bar(
        live_plan,
        x="hr",
        y=["prediction", "planned_availability"],
        barmode="group",
        title="Forecast vs planned availability (hourly)",
    )
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    move_matrix = zone_move_plan.pivot(index="origin", columns="destination", values="bikes_moved").fillna(0)
    fig = px.imshow(
        move_matrix,
        text_auto=True,
        aspect="auto",
        title="Overnight zone move matrix",
        color_continuous_scale=[[0, "#0f172a"], [0.5, "#134e4a"], [1, "#2dd4bf"]],
    )
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(zone_summary, use_container_width=True)

with tabs[3]:
    st.markdown(
        '<p class="velo-section-title">Scenario testing</p><p class="velo-section-hint">Compare precomputed operational scenarios on cost index and fleet movement.</p>',
        unsafe_allow_html=True,
    )
    scenario_results = artifacts["scenario_results"].copy()
    fig = px.bar(scenario_results, x="scenario", y="weighted_cost_index", color="scenario", title="Weighted cost index by scenario")
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)
    fig = px.bar(scenario_results, x="scenario", y="bikes_moved", color="scenario", title="Bikes moved by scenario")
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(scenario_results.sort_values("weighted_cost_index"), use_container_width=True)

with tabs[4]:
    st.markdown(
        '<p class="velo-section-title">Model diagnostics</p><p class="velo-section-hint">Holdout errors and feature contributions from the training run.</p>',
        unsafe_allow_html=True,
    )
    overall_metrics = artifacts["holdout_metrics"].loc[artifacts["holdout_metrics"]["segment_type"] == "overall"].copy()
    fig = px.bar(overall_metrics, x="model", y="rmse", title="Holdout RMSE by model")
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)
    fig = px.bar(overall_metrics, x="model", y="mae", title="Holdout MAE by model")
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    feature_importance = artifacts["feature_importance"].copy()
    if not feature_importance.empty:
        top_features = feature_importance.sort_values("importance_mean", ascending=False).head(15)
        fig = px.bar(top_features, x="importance_mean", y="feature", orientation="h", title="Top feature importances")
        style_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.markdown(
        '<p class="velo-section-title">Business impact</p><p class="velo-section-hint">Forecast-driven planning vs static baseline: cost structure and service proxies.</p>',
        unsafe_allow_html=True,
    )
    policy_summary = artifacts["policy_cost_breakdown"].copy()
    st.dataframe(policy_summary, use_container_width=True)
    forecast_driven = policy_summary.loc[policy_summary["policy"] == "forecast_driven"].iloc[0]
    baseline = policy_summary.loc[policy_summary["policy"] == "static_hourly_mean"].iloc[0]
    improvement = baseline["weighted_cost_index"] - forecast_driven["weighted_cost_index"]

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Weighted cost improvement", f"{improvement:,.1f}")
    kpi_cols[1].metric("Service level proxy", f"{forecast_driven['mean_service_level_proxy']:.2%}")
    kpi_cols[2].metric("Utilization proxy", f"{forecast_driven['mean_utilization_proxy']:.2%}")
    kpi_cols[3].metric("Baseline cost index", f"{baseline['weighted_cost_index']:,.1f}")

    fig = px.bar(
        policy_summary,
        x="policy",
        y=["total_proxy_shortage", "total_proxy_idle"],
        barmode="group",
        title="Shortage vs idle totals: baseline vs forecast-driven",
    )
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)
