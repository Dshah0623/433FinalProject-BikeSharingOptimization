from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bikeshare_project.config import load_config
from bikeshare_project.io_utils import save_dataframe, save_joblib, save_json
from bikeshare_project.models import (
    select_best_model,
    summarize_cv_metrics,
    summarize_feature_importance,
    summarize_holdout_predictions,
    train_and_evaluate_models,
)
from bikeshare_project.paths import DATA_PROCESSED, MODELS, RESULTS_TABLES


def main() -> None:
    config = load_config()
    model_df = pd.read_parquet(DATA_PROCESSED / "model_table.parquet")
    results = train_and_evaluate_models(model_df, config)
    best_model_name = select_best_model(results)
    best_result = results[best_model_name]

    cv_metrics = summarize_cv_metrics(results)
    holdout_predictions = summarize_holdout_predictions(results)
    feature_importance = summarize_feature_importance(results)
    holdout_metrics = pd.concat([result.holdout_metrics for result in results.values()], ignore_index=True)
    cv_predictions = pd.concat(
        [result.cv_predictions.assign(model=result.name) for result in results.values()],
        ignore_index=True,
    )

    save_dataframe(cv_metrics, RESULTS_TABLES / "cv_metrics.csv")
    save_dataframe(holdout_predictions, RESULTS_TABLES / "holdout_predictions.csv")
    save_dataframe(holdout_metrics, RESULTS_TABLES / "holdout_metrics.csv")
    save_dataframe(feature_importance, RESULTS_TABLES / "feature_importance.csv")
    save_dataframe(cv_predictions, DATA_PROCESSED / "cv_predictions.csv")
    if best_result.estimator is not None:
        save_joblib(best_result.estimator, MODELS / "best_model.joblib")
    save_json(
        {
            "best_model_name": best_model_name,
            "best_params": best_result.best_params,
            "all_best_params": {name: result.best_params for name, result in results.items()},
        },
        MODELS / "best_model_metadata.json",
    )
    print(f"Saved model results. Best model: {best_model_name}")


if __name__ == "__main__":
    main()
