"""
Plotly Dash dashboard — same analytics as the Streamlit app, without Streamlit.

Run from project root:
  python dashboard/dash_app.py
Then open http://127.0.0.1:8050
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc

DASHBOARD_DIR = Path(__file__).resolve().parent
ROOT = DASHBOARD_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(DASHBOARD_DIR))

from bikeshare_project.config import load_config
from bikeshare_project.optimization import build_availability_plan, derive_zone_targets, solve_zone_rebalancing
from bikeshare_project.paths import DATA_PROCESSED, MODELS, RESULTS_RECOMMENDATIONS, RESULTS_TABLES

# Light-theme chart palette: saturated hues readable on white / slate-50 plot areas
TEXT_PRIMARY = "#0f172a"  # slate-900 — body & titles (WCAG on white)
TEXT_SECONDARY = "#475569"  # slate-600 — ticks, legend
PLOT_BG = "#f8fafc"
COLORWAY = ["#4f46e5", "#b45309", "#be185d", "#047857", "#0369a1", "#c2410c"]

# Forecast traces: strong contrast on light plot background
DASH_LINE = "#3730a3"  # indigo-800
DASH_FILL = "rgba(55, 48, 163, 0.18)"
DASH_ACTUAL = "#9d174d"  # pink-800


def style_plotly_light(fig: go.Figure) -> go.Figure:
    """Plotly styling for light UI: dark text, light plot area, subtle grids."""
    grid = "rgba(15, 23, 42, 0.1)"
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(family="Outfit, sans-serif", color=TEXT_PRIMARY, size=12),
        title_font=dict(size=15, color=TEXT_PRIMARY),
        margin=dict(l=52, r=28, t=52, b=48),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(15,23,42,0.12)",
            borderwidth=1,
            font=dict(color=TEXT_PRIMARY, size=11),
        ),
        xaxis=dict(
            gridcolor=grid,
            zerolinecolor=grid,
            tickfont=dict(color=TEXT_SECONDARY),
            title_font=dict(color=TEXT_PRIMARY),
            linecolor="rgba(15,23,42,0.2)",
        ),
        yaxis=dict(
            gridcolor=grid,
            zerolinecolor=grid,
            tickfont=dict(color=TEXT_SECONDARY),
            title_font=dict(color=TEXT_PRIMARY),
            linecolor="rgba(15,23,42,0.2)",
        ),
        colorway=COLORWAY,
    )
    return fig


def finalize_fig(fig: go.Figure) -> go.Figure:
    """Dash light theme for all Plotly figures."""
    style_plotly_light(fig)
    return fig


def tab_guide(title: str, markdown_body: str) -> dbc.Card:
    """Short note above tab content (sources and behavior)."""
    return dbc.Card(
        [
            html.Div([html.Span("ⓘ", className="me-1"), title], className="guide-title"),
            dcc.Markdown(markdown_body, className="guide-md", dangerously_allow_html=False),
        ],
        className="guide-card mb-3",
        body=True,
    )


def load_artifacts() -> dict:
    config = load_config()
    clean_hourly = pd.read_csv(DATA_PROCESSED / "clean_hourly.csv", parse_dates=["dteday", "timestamp", "date"])
    model_table = pd.read_parquet(DATA_PROCESSED / "model_table.parquet")
    next_day_forecasts = pd.read_csv(DATA_PROCESSED / "next_day_forecasts.csv", parse_dates=["timestamp", "date"])
    policy_cost_breakdown = pd.read_csv(RESULTS_RECOMMENDATIONS / "policy_cost_breakdown.csv")
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
        "policy_cost_breakdown": policy_cost_breakdown,
        "scenario_results": scenario_results,
        "cv_predictions": cv_predictions,
        "holdout_metrics": holdout_metrics,
        "feature_importance": feature_importance,
        "metadata": metadata,
    }


artifacts = load_artifacts()
config = artifacts["config"]
opt = config["optimization"]


def recalculate_live_plan(shortage_cost: float, idle_cost: float, fleet_size: int) -> pd.DataFrame:
    best_model_name = artifacts["metadata"]["best_model_name"]
    best_cv = artifacts["cv_predictions"].loc[artifacts["cv_predictions"]["model"] == best_model_name].copy()
    residuals = best_cv["cnt"] - best_cv["prediction"]
    return build_availability_plan(
        forecast_df=artifacts["next_day_forecasts"],
        residuals=residuals,
        shortage_cost=shortage_cost,
        idle_cost=idle_cost,
        fleet_size=float(fleet_size),
    )


def fig_historical(include_anomaly: bool) -> tuple[go.Figure, go.Figure, go.Figure]:
    clean_hourly = artifacts["clean_hourly"].copy()
    if not include_anomaly:
        clean_hourly = clean_hourly.loc[clean_hourly["is_anomaly_day"] == 0]
    daily_totals = clean_hourly.groupby("date", as_index=False)["cnt"].sum()
    fig1 = px.line(daily_totals, x="date", y="cnt", title="Daily total demand")
    finalize_fig(fig1)

    hourly_profile = clean_hourly.groupby(["workingday", "hr"], as_index=False)["cnt"].mean()
    hourly_profile["day_type"] = hourly_profile["workingday"].map({1: "Working day", 0: "Weekend/Holiday"})
    fig2 = px.line(hourly_profile, x="hr", y="cnt", color="day_type", title="Hourly profile by day type")
    finalize_fig(fig2)

    weather_profile = clean_hourly.groupby("weathersit", as_index=False)["cnt"].mean()
    fig3 = px.bar(weather_profile, x="weathersit", y="cnt", title="Average demand by weather code")
    finalize_fig(fig3)
    return fig1, fig2, fig3


def fig_forecast() -> go.Figure:
    forecast_df = artifacts["next_day_forecasts"].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["prediction"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=DASH_LINE, width=2.4),
            marker=dict(size=5, color=DASH_LINE),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(forecast_df["timestamp"]) + list(forecast_df["timestamp"][::-1]),
            y=list(forecast_df["prediction_upper"]) + list(forecast_df["prediction_lower"][::-1]),
            fill="toself",
            fillcolor=DASH_FILL,
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
                line=dict(dash="dash", color=DASH_ACTUAL, width=1.8),
            )
        )
    fig.update_layout(title="Reference next-day forecast")
    finalize_fig(fig)
    return fig


def build_allocation_figs(
    shortage_cost: float,
    idle_cost: float,
    fleet_size: int,
    move_cost: float,
    residential_share: float,
    employment_share: float,
    inv_r: int,
    inv_e: int,
    inv_l: int,
) -> tuple[dbc.Row, go.Figure, go.Figure, dbc.Table, str]:
    leisure_share = max(0.0, round(1.0 - residential_share - employment_share, 2))
    zone_shares = {
        "Residential": residential_share,
        "Employment Core": employment_share,
        "Leisure/Mixed Use": leisure_share,
    }
    current_inventory = {
        "Residential": inv_r,
        "Employment Core": inv_e,
        "Leisure/Mixed Use": inv_l,
    }
    live_plan = recalculate_live_plan(shortage_cost, idle_cost, int(fleet_size))
    zone_targets = derive_zone_targets(live_plan, config, zone_shares=zone_shares)
    zone_summary, zone_move_plan = solve_zone_rebalancing(
        zone_targets=zone_targets,
        current_inventory=current_inventory,
        move_cost=move_cost,
        shortage_cost=shortage_cost,
        idle_cost=idle_cost,
        move_capacity=int(opt["move_capacity"]),
    )

    metrics = dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [
                        html.Div("Planned availability total", className="metric-label"),
                        html.Div(f"{live_plan['planned_availability'].sum():,.0f}", className="metric-value"),
                    ],
                    className="metric-card",
                ),
                width=6,
                md=3,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.Div("Proxy shortage", className="metric-label"),
                        html.Div(f"{live_plan['proxy_shortage'].sum():,.0f}", className="metric-value"),
                    ],
                    className="metric-card",
                ),
                width=6,
                md=3,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.Div("Proxy idle", className="metric-label"),
                        html.Div(f"{live_plan['proxy_idle'].sum():,.0f}", className="metric-value"),
                    ],
                    className="metric-card",
                ),
                width=6,
                md=3,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.Div("Bikes moved", className="metric-label"),
                        html.Div(f"{zone_move_plan['bikes_moved'].sum():,.0f}", className="metric-value"),
                    ],
                    className="metric-card",
                ),
                width=6,
                md=3,
            ),
        ],
        className="g-2 mb-3",
    )

    fig_bar = px.bar(
        live_plan,
        x="hr",
        y=["prediction", "planned_availability"],
        barmode="group",
        title="Forecast vs planned availability (hourly)",
    )
    finalize_fig(fig_bar)

    move_matrix = zone_move_plan.pivot(index="origin", columns="destination", values="bikes_moved").fillna(0)
    fig_heat = px.imshow(
        move_matrix,
        text_auto=True,
        aspect="auto",
        title="Overnight zone move matrix",
        # Stay in mid-tones so cell labels stay readable with dark text (single textfont color)
        color_continuous_scale=[
            [0, "#f8fafc"],
            [0.4, "#e0e7ff"],
            [0.75, "#a5b4fc"],
            [1, "#6366f1"],
        ],
    )
    finalize_fig(fig_heat)
    fig_heat.update_traces(textfont=dict(color=TEXT_PRIMARY, size=11))
    fig_heat.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            outlinewidth=1,
            outlinecolor="rgba(15,23,42,0.12)",
        )
    )

    table = dbc.Table.from_dataframe(
        zone_summary,
        striped=True,
        bordered=True,
        hover=True,
        className="table-light table-bordered table-ops table-sm mb-0",
    )
    leisure_text = f"Leisure / Mixed Use share: {leisure_share:.2f} (remainder after Residential + Employment)"
    return metrics, fig_bar, fig_heat, table, leisure_text


# Initial figures for static tabs
_f_hist = fig_historical(False)
_forecast_fig = fig_forecast()
_scenario = artifacts["scenario_results"].copy()
_fig_scen_cost = px.bar(_scenario, x="scenario", y="weighted_cost_index", color="scenario", title="Weighted cost index by scenario")
finalize_fig(_fig_scen_cost)
_fig_scen_move = px.bar(_scenario, x="scenario", y="bikes_moved", color="scenario", title="Bikes moved by scenario")
finalize_fig(_fig_scen_move)

_overall = artifacts["holdout_metrics"].loc[artifacts["holdout_metrics"]["segment_type"] == "overall"].copy()
_fig_rmse = px.bar(_overall, x="model", y="rmse", title="Holdout RMSE by model")
finalize_fig(_fig_rmse)
_fig_mae = px.bar(_overall, x="model", y="mae", title="Holdout MAE by model")
finalize_fig(_fig_mae)

_fi = artifacts["feature_importance"]
if not _fi.empty:
    _top = _fi.sort_values("importance_mean", ascending=False).head(15)
    _fig_fi = px.bar(_top, x="importance_mean", y="feature", orientation="h", title="Top feature importances")
    finalize_fig(_fig_fi)
else:
    _fig_fi = go.Figure()
    _fig_fi.update_layout(title=dict(text="No feature importance (naive model)", font=dict(color=TEXT_PRIMARY)))
    finalize_fig(_fig_fi)

_policy = artifacts["policy_cost_breakdown"].copy()
_fig_policy = px.bar(
    _policy,
    x="policy",
    y=["total_proxy_shortage", "total_proxy_idle"],
    barmode="group",
    title="Shortage vs idle totals: baseline vs forecast-driven",
)
finalize_fig(_fig_policy)

_fd = _policy.loc[_policy["policy"] == "forecast_driven"].iloc[0]
_bl = _policy.loc[_policy["policy"] == "static_hourly_mean"].iloc[0]
_improvement = _bl["weighted_cost_index"] - _fd["weighted_cost_index"]

forecast_df = artifacts["next_day_forecasts"]
_init_metrics, _init_bar, _init_heat, _init_table, _init_leisure = build_allocation_figs(
    float(opt["shortage_cost"]),
    float(opt["idle_cost"]),
    int(opt["total_fleet_size"]),
    float(opt["move_cost"]),
    float(opt["default_zone_shares"]["workingday"]["Residential"]),
    float(opt["default_zone_shares"]["workingday"]["Employment Core"]),
    int(opt["default_current_inventory"]["Residential"]),
    int(opt["default_current_inventory"]["Employment Core"]),
    int(opt["default_current_inventory"]["Leisure/Mixed Use"]),
)

# Short tab notes (data source, limits)
_GUIDE_HIST = """
Daily system-wide trips, mean demand by hour (working day vs weekend/holiday), and mean demand by weather code, from `clean_hourly`. Anomaly days follow the validation rules (z-scores and configured dates). Toggle *Include anomaly days* to include or exclude those dates.
"""

_GUIDE_FC = """
Reference next-day hourly forecast for the selected day and an empirical band from cross-validation residuals; actuals appear when the day is in history. Values come from `05_generate_forecasts` and do not respond to sidebar costs (those affect the Allocation tab only).
"""

_GUIDE_ALLOC = """
Planned hourly availability vs raw forecast, shortage/idle proxies, moves, zone matrix, and post-LP inventory. A critical fractile on CV residuals feeds a newsvendor-style plan clipped to fleet size; zone splits use the share sliders; PuLP minimizes move and imbalance costs. Adjust sliders to refresh charts and tables.
"""

_GUIDE_SCEN = """
Batch scenario rows from `08_scenarios` (weather stress, surge, forecast error scale, fleet, costs), each re-optimized for availability and zones, sorted by weighted cost index. Rerun the pipeline to change scenarios; use Allocation for interactive what-ifs.
"""

_GUIDE_DIAG = """
Holdout RMSE/MAE by model and permutation importance for the selected model (importance empty for seasonal naive). Metrics use the late-2012 holdout with optional anomaly exclusion. Retrain via `04_train_models` through the full pipeline.
"""

_GUIDE_BIZ = """
Forecast-based policy vs a static hourly-mean baseline: proxy shortage/idle, cost index, and service/utilization proxies for the reference run from `06_capacity_policy`. Figures do not follow sidebar sliders; compare to the Allocation tab for live cost changes.
"""

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    assets_folder=str(DASHBOARD_DIR / "assets"),
    suppress_callback_exceptions=True,
)
server = app.server

app.title = "Bike-share dashboard (Dash)"

app.layout = dbc.Container(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div("MSE 433", className="velo-kicker"),
                        html.H1("Demand, availability, and zone rebalancing", className="velo-title"),
                        html.P(
                            [
                                "Tabs cover history, forecast, interactive allocation, scenarios, diagnostics, and policy comparison. The left panel updates ",
                                html.Strong("allocation"),
                                " and rebalancing; each tab opens with a short note on data sources and limits.",
                            ],
                            className="velo-sub",
                        ),
                        html.Div(
                            [
                                html.Span(f"Best model · {artifacts['metadata']['best_model_name']}", className="velo-pill"),
                                html.Span("Newsvendor + zone LP", className="velo-pill"),
                            ],
                            className="velo-meta",
                        ),
                    ],
                    className="velo-hero-inner",
                ),
            ],
            className="velo-hero",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Div("Bike-share", className="sidebar-brand"),
                                html.Div("Costs and zones", className="sidebar-title"),
                                html.Label("Shortage penalty", className="label-ops small"),
                                dcc.Slider(
                                    id="shortage-cost",
                                    min=1,
                                    max=12,
                                    step=0.5,
                                    value=float(opt["shortage_cost"]),
                                    marks=None,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                                html.Label("Idle cost", className="label-ops small mt-2"),
                                dcc.Slider(
                                    id="idle-cost",
                                    min=0.5,
                                    max=5,
                                    step=0.5,
                                    value=float(opt["idle_cost"]),
                                ),
                                html.Label("Move cost", className="label-ops small mt-2"),
                                dcc.Slider(
                                    id="move-cost",
                                    min=0.1,
                                    max=3,
                                    step=0.1,
                                    value=float(opt["move_cost"]),
                                ),
                                html.Label("Fleet size", className="label-ops small mt-2"),
                                dcc.Slider(
                                    id="fleet-size",
                                    min=500,
                                    max=1300,
                                    step=25,
                                    value=int(opt["total_fleet_size"]),
                                ),
                                html.Hr(className="sidebar-rule my-3"),
                                html.Label("Current inventory — Residential", className="label-ops small"),
                                dbc.Input(id="inv-r", type="number", min=0, value=int(opt["default_current_inventory"]["Residential"]), className="mb-2 form-control-ops"),
                                html.Label("Employment Core", className="label-ops small"),
                                dbc.Input(id="inv-e", type="number", min=0, value=int(opt["default_current_inventory"]["Employment Core"]), className="mb-2 form-control-ops"),
                                html.Label("Leisure / Mixed Use", className="label-ops small"),
                                dbc.Input(id="inv-l", type="number", min=0, value=int(opt["default_current_inventory"]["Leisure/Mixed Use"]), className="mb-2 form-control-ops"),
                                html.Hr(className="sidebar-rule my-3"),
                                html.P("Zone share assumptions", className="label-ops small fw-bold mb-2"),
                                html.Label("Residential share", className="label-ops small"),
                                dcc.Slider(id="share-r", min=0, max=1, step=0.01, value=float(opt["default_zone_shares"]["workingday"]["Residential"])),
                                html.Label("Employment Core share", className="label-ops small mt-2"),
                                dcc.Slider(id="share-e", min=0, max=1, step=0.01, value=float(opt["default_zone_shares"]["workingday"]["Employment Core"])),
                                html.P(id="leisure-label", children=_init_leisure, className="leisure-hint mt-2 mb-0"),
                            ],
                            className="sidebar-panel",
                        ),
                    ],
                    width=12,
                    lg=3,
                    className="sidebar-sticky",
                ),
                dbc.Col(
                    [
                        dcc.Tabs(
                            id="main-tabs",
                            className="dash-tabs",
                            value="tab-alloc",
                            children=[
                                dcc.Tab(
                                    label="Historical",
                                    value="tab-hist",
                                    children=[
                                        tab_guide("Historical — demand context", _GUIDE_HIST),
                                        dbc.Switch(id="anomaly-switch", label="Include anomaly days in charts", value=False, className="mb-3 switch-ops"),
                                        dcc.Graph(id="hist-daily", figure=_f_hist[0], config={"displayModeBar": True}),
                                        dcc.Graph(id="hist-hourly", figure=_f_hist[1]),
                                        dcc.Graph(id="hist-weather", figure=_f_hist[2]),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Forecast",
                                    value="tab-fc",
                                    children=[
                                        tab_guide("Forecast — next-day trajectory", _GUIDE_FC),
                                        dcc.Graph(figure=_forecast_fig, config={"displayModeBar": True}),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([html.Div("Forecast total", className="metric-label"), html.Div(f"{forecast_df['prediction'].sum():,.0f}", className="metric-value")], className="metric-card"), width=6, md=3),
                                                dbc.Col(html.Div([html.Div("Peak-hour", className="metric-label"), html.Div(f"{forecast_df['prediction'].max():,.0f}", className="metric-value")], className="metric-card"), width=6, md=3),
                                                dbc.Col(html.Div([html.Div("Mean", className="metric-label"), html.Div(f"{forecast_df['prediction'].mean():.1f}", className="metric-value")], className="metric-card"), width=6, md=3),
                                                dbc.Col(html.Div([html.Div("Model", className="metric-label"), html.Div(artifacts["metadata"]["best_model_name"], className="metric-value", style={"fontSize": "1rem"})], className="metric-card"), width=6, md=3),
                                            ],
                                            className="g-2 mt-2",
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Allocation",
                                    value="tab-alloc",
                                    children=[
                                        tab_guide("Allocation — live optimization", _GUIDE_ALLOC),
                                        html.Div(id="alloc-metrics", children=_init_metrics),
                                        dcc.Graph(id="alloc-bar", figure=_init_bar),
                                        dcc.Graph(id="alloc-heat", figure=_init_heat),
                                        html.Div(id="alloc-table", children=_init_table),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Scenarios",
                                    value="tab-scen",
                                    children=[
                                        tab_guide("Scenarios — batch what-ifs", _GUIDE_SCEN),
                                        dcc.Graph(figure=_fig_scen_cost),
                                        dcc.Graph(figure=_fig_scen_move),
                                        dbc.Table.from_dataframe(
                                            _scenario.sort_values("weighted_cost_index"),
                                            striped=True,
                                            bordered=True,
                                            hover=True,
                                            className="table-light table-bordered table-ops table-sm",
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Diagnostics",
                                    value="tab-diag",
                                    children=[
                                        tab_guide("Diagnostics — model quality", _GUIDE_DIAG),
                                        dcc.Graph(figure=_fig_rmse),
                                        dcc.Graph(figure=_fig_mae),
                                        dcc.Graph(figure=_fig_fi),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Business impact",
                                    value="tab-biz",
                                    children=[
                                        tab_guide("Business impact — policy comparison", _GUIDE_BIZ),
                                        dbc.Table.from_dataframe(_policy, striped=True, bordered=True, hover=True, className="table-light table-bordered table-ops table-sm mb-3"),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([html.Div("Cost improvement", className="metric-label"), html.Div(f"{_improvement:,.1f}", className="metric-value")], className="metric-card"), width=6, md=3),
                                                dbc.Col(html.Div([html.Div("Service level proxy", className="metric-label"), html.Div(f"{_fd['mean_service_level_proxy']:.2%}", className="metric-value")], className="metric-card"), width=6, md=3),
                                                dbc.Col(html.Div([html.Div("Utilization proxy", className="metric-label"), html.Div(f"{_fd['mean_utilization_proxy']:.2%}", className="metric-value")], className="metric-card"), width=6, md=3),
                                                dbc.Col(html.Div([html.Div("Baseline cost index", className="metric-label"), html.Div(f"{_bl['weighted_cost_index']:,.1f}", className="metric-value")], className="metric-card"), width=6, md=3),
                                            ],
                                            className="g-2 mb-3",
                                        ),
                                        dcc.Graph(figure=_fig_policy),
                                    ],
                                ),
                            ],
                        ),
                    ],
                    width=12,
                    lg=9,
                ),
            ],
            className="g-4",
        ),
    ],
    fluid=True,
    className="dash-container py-3",
)


@callback(
    Output("hist-daily", "figure"),
    Output("hist-hourly", "figure"),
    Output("hist-weather", "figure"),
    Input("anomaly-switch", "value"),
)
def update_historical(include_anomaly: bool):
    return fig_historical(bool(include_anomaly))


@callback(
    Output("alloc-metrics", "children"),
    Output("alloc-bar", "figure"),
    Output("alloc-heat", "figure"),
    Output("alloc-table", "children"),
    Output("leisure-label", "children"),
    Input("shortage-cost", "value"),
    Input("idle-cost", "value"),
    Input("move-cost", "value"),
    Input("fleet-size", "value"),
    Input("share-r", "value"),
    Input("share-e", "value"),
    Input("inv-r", "value"),
    Input("inv-e", "value"),
    Input("inv-l", "value"),
)
def update_allocation(shortage, idle, move_c, fleet, sr, se, ir, ie, il):
    shortage = float(shortage)
    idle = float(idle)
    move_c = float(move_c)
    fleet = int(fleet) if fleet is not None else int(opt["total_fleet_size"])
    sr = float(sr) if sr is not None else 0.42
    se = float(se) if se is not None else 0.43
    ir = int(ir) if ir is not None else 0
    ie = int(ie) if ie is not None else 0
    il = int(il) if il is not None else 0
    m, fb, fh, tbl, leisure = build_allocation_figs(shortage, idle, fleet, move_c, sr, se, ir, ie, il)
    return m, fb, fh, tbl, leisure


if __name__ == "__main__":
    app.run(debug=True, port=8050)
