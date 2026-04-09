# Bike-Share Forecast, Allocate, Rebalance

This repository implements an end-to-end decision-support system for the MSE 433 individual final project using the Capital Bikeshare hourly demand dataset. The workflow connects:

1. data validation and anomaly review
2. descriptive analytics
3. next-day hourly demand forecasting
4. cost-based availability planning
5. stylized overnight zone rebalancing
6. scenario testing
7. dashboards: **Streamlit** (`dashboard/app.py`) or **Plotly Dash** (`dashboard/dash_app.py`)

## Project Structure

```text
configs/                  Project configuration
data/raw/                 Copied source data
data/interim/             Validation and EDA artifacts
data/processed/           Modeling tables and forecast artifacts
dashboard/                Streamlit + Dash apps, shared `theme.py`
models/                   Saved trained models and metadata
reports/                  Report outline and draft-ready content
results/figures/          Saved report figures
results/tables/           Saved analysis tables
results/recommendations/  Saved policy outputs
src/bikeshare_project/    Shared package code
01_validate_data.py       Data validation pipeline stage
02_eda.py                 Descriptive analytics stage
03_build_features.py      Feature engineering stage
04_train_models.py        Forecast model training and evaluation
05_generate_forecasts.py  Final forecast generation
06_capacity_policy.py     Availability planning stage
07_zone_rebalancing.py    Overnight zone balancing stage
08_scenarios.py           Scenario and sensitivity analysis stage
run_pipeline.py           Full end-to-end execution
```

## Data Inputs

The default configuration expects the provided source files at:

- `/Users/devshah/Downloads/archive/hour.csv`
- `/Users/devshah/Downloads/archive/day.csv`
- `/Users/devshah/Downloads/archive/Readme.txt`

`01_validate_data.py` copies the CSV files into `data/raw/` so the rest of the pipeline uses workspace-local inputs.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run dashboard/app.py
# optional — same analytics without Streamlit:
python dashboard/dash_app.py   # http://127.0.0.1:8050
```

## Key Modeling Choices

- Forecast target: hourly total rentals `cnt`
- Forecast horizon: next 24 hours, generated at end-of-day
- Holdout design: late-2012 complete days, with explicit anomaly handling for October 29, 2012
- Forecast stack: seasonal naive baseline, regularized linear regression, Poisson benchmark, boosted trees
- Core prescription: critical-fractile hourly availability targets
- Spatial prototype: configurable 3-zone overnight rebalancing model

## Academic Guardrails

- The project makes no station-level claims because the dataset is system-level, not station-level.
- Rebalancing is presented as a manager-configurable zone prototype, not as empirical station-to-station routing.
- Same-period `casual` and `registered` values are excluded from forecasting features because they would leak the target.
