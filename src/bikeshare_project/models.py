from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import warnings
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import get_feature_columns
from .metrics import regression_metrics, segmented_metrics


warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ModelResult:
    name: str
    best_params: Dict[str, Any]
    estimator: Any
    cv_predictions: pd.DataFrame
    cv_metrics: pd.DataFrame
    holdout_predictions: pd.DataFrame
    holdout_metrics: pd.DataFrame
    feature_importance: pd.DataFrame


def _build_linear_pipeline(config: Dict[str, Any], alpha: float) -> Pipeline:
    del config
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _build_poisson_pipeline(config: Dict[str, Any], alpha: float) -> Pipeline:
    del config
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=alpha, max_iter=1000)),
        ]
    )


def _build_boosting_pipeline(config: Dict[str, Any], learning_rate: float, max_depth: int, max_iter: int) -> Pipeline:
    del config
    return Pipeline(
        steps=[
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    max_iter=max_iter,
                    random_state=42,
                    loss="squared_error",
                ),
            ),
        ]
    )


def _create_cv_folds(df: pd.DataFrame, config: Dict[str, Any]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    holdout_start = pd.Timestamp(config["modeling"]["holdout_start_date"])
    complete_train_dates = sorted(df.loc[df["date"] < holdout_start, "date"].drop_duplicates())
    n_folds = int(config["modeling"]["cv"]["n_folds"])
    validation_days = int(config["modeling"]["cv"]["validation_days"])
    min_train_days = int(config["modeling"]["cv"]["min_train_days"])

    folds: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for fold_idx in range(n_folds, 0, -1):
        val_end_idx = len(complete_train_dates) - validation_days * (fold_idx - 1)
        val_start_idx = val_end_idx - validation_days
        if val_start_idx < min_train_days:
            continue
        val_dates = complete_train_dates[val_start_idx:val_end_idx]
        folds.append((val_dates[0], val_dates[-1]))
    return folds


def _train_validation_split(
    df: pd.DataFrame,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[df["date"] < val_start].copy()
    val_df = df.loc[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
    return train_df, val_df


def _prepare_xy(df: pd.DataFrame, config: Dict[str, Any], target_mode: str) -> Tuple[pd.DataFrame, np.ndarray]:
    features = df[get_feature_columns(config)].copy()
    if target_mode == "log1p":
        target = df["target_log1p"].to_numpy()
    else:
        target = df["cnt"].to_numpy()
    return features, target


def _predict(estimator: Any, x: pd.DataFrame, target_mode: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        preds = estimator.predict(x)
    if target_mode == "log1p":
        preds = np.expm1(np.clip(preds, a_min=-20, a_max=10))
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(preds, a_min=0, a_max=None)


def _seasonal_naive_predict(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["lag_168"].notna(), df["lag_168"], df["lag_24"])


def _evaluate_predictions(scored_df: pd.DataFrame, model_name: str, split_name: str) -> pd.DataFrame:
    overall = regression_metrics(scored_df["cnt"].to_numpy(), scored_df["prediction"].to_numpy())
    overall.update({"model": model_name, "split": split_name, "segment_type": "overall", "segment_value": "all"})
    rows = [overall]

    for segment_col in ["hr", "workingday", "season", "weathersit"]:
        segment_df = segmented_metrics(scored_df, "cnt", "prediction", segment_col, prefix=segment_col)
        if segment_df.empty:
            continue
        segment_df = segment_df.rename(columns={segment_col: "segment_value"})
        segment_df["model"] = model_name
        segment_df["split"] = split_name
        rows.extend(segment_df.to_dict(orient="records"))

    peak_threshold = scored_df["cnt"].quantile(0.9)
    peak_df = scored_df.loc[scored_df["cnt"] >= peak_threshold]
    if not peak_df.empty:
        peak_metrics = regression_metrics(peak_df["cnt"].to_numpy(), peak_df["prediction"].to_numpy())
        peak_metrics.update({"model": model_name, "split": split_name, "segment_type": "peak_demand", "segment_value": "top_decile"})
        rows.append(peak_metrics)

    return pd.DataFrame(rows)


def _permutation_feature_importance(estimator: Any, scoring_df: pd.DataFrame, config: Dict[str, Any], model_name: str) -> pd.DataFrame:
    if model_name == "seasonal_naive":
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    x = scoring_df[get_feature_columns(config)]
    y = scoring_df["cnt"].to_numpy()
    result = permutation_importance(estimator, x, y, n_repeats=3, random_state=42, scoring="neg_root_mean_squared_error")
    return (
        pd.DataFrame(
            {
                "feature": get_feature_columns(config),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _score_naive(train_df: pd.DataFrame, val_df: pd.DataFrame, _: Dict[str, Any]) -> np.ndarray:
    del train_df
    return _seasonal_naive_predict(val_df)


def _fit_generic_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict[str, Any],
    builder,
    params: Dict[str, Any],
    target_mode: str,
) -> Tuple[Any, np.ndarray]:
    estimator = builder(config, **params)
    x_train, y_train = _prepare_xy(train_df, config, target_mode)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        estimator.fit(x_train, y_train)
    val_predictions = _predict(estimator, val_df[get_feature_columns(config)], target_mode)
    return estimator, val_predictions


def _candidate_configs() -> Dict[str, Iterable[Dict[str, Any]]]:
    return {
        "linear_ridge": [{"alpha": alpha} for alpha in [1.0, 10.0]],
        "poisson": [{"alpha": alpha} for alpha in [0.01, 0.1]],
        "boosted_tree": [
            {"learning_rate": lr, "max_depth": depth, "max_iter": max_iter}
            for lr in [0.05]
            for depth in [4, 6]
            for max_iter in [250]
        ],
    }


def train_and_evaluate_models(model_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, ModelResult]:
    folds = _create_cv_folds(model_df, config)
    holdout_start = pd.Timestamp(config["modeling"]["holdout_start_date"])
    holdout_df = model_df.loc[model_df["date"] >= holdout_start].copy()
    anomaly_dates = {pd.Timestamp(value) for value in config["modeling"]["anomaly_dates"]}
    if not config["modeling"]["include_anomaly_in_primary_holdout"]:
        holdout_df = holdout_df.loc[~holdout_df["date"].isin(anomaly_dates)].copy()

    results: Dict[str, ModelResult] = {}

    naive_cv_rows = []
    naive_cv_predictions = []
    for fold_idx, (val_start, val_end) in enumerate(folds, start=1):
        train_df, val_df = _train_validation_split(model_df, val_start, val_end)
        preds = _score_naive(train_df, val_df, config)
        scored = val_df.copy()
        scored["prediction"] = preds
        scored["fold"] = fold_idx
        naive_cv_predictions.append(scored)
        fold_metrics = regression_metrics(scored["cnt"].to_numpy(), scored["prediction"].to_numpy())
        fold_metrics.update({"model": "seasonal_naive", "fold": fold_idx, "params": "{}"})
        naive_cv_rows.append(fold_metrics)

    naive_cv_predictions_df = pd.concat(naive_cv_predictions, ignore_index=True)
    naive_holdout = holdout_df.copy()
    naive_holdout["prediction"] = _seasonal_naive_predict(naive_holdout)
    results["seasonal_naive"] = ModelResult(
        name="seasonal_naive",
        best_params={},
        estimator=None,
        cv_predictions=naive_cv_predictions_df,
        cv_metrics=pd.DataFrame(naive_cv_rows),
        holdout_predictions=naive_holdout,
        holdout_metrics=_evaluate_predictions(naive_holdout, "seasonal_naive", "holdout"),
        feature_importance=pd.DataFrame(columns=["feature", "importance_mean", "importance_std"]),
    )

    builders = {
        "linear_ridge": (_build_linear_pipeline, "log1p"),
        "poisson": (_build_poisson_pipeline, "count"),
        "boosted_tree": (_build_boosting_pipeline, "count"),
    }

    for model_name, candidate_params in _candidate_configs().items():
        builder, target_mode = builders[model_name]
        fold_scores = []
        for params in candidate_params:
            params_key = str(params)
            for fold_idx, (val_start, val_end) in enumerate(folds, start=1):
                train_df, val_df = _train_validation_split(model_df, val_start, val_end)
                estimator, preds = _fit_generic_model(train_df, val_df, config, builder, params, target_mode)
                scored = val_df.copy()
                scored["prediction"] = preds
                metrics = regression_metrics(scored["cnt"].to_numpy(), scored["prediction"].to_numpy())
                metrics.update({"model": model_name, "fold": fold_idx, "params": params_key})
                fold_scores.append(metrics)
        fold_scores_df = pd.DataFrame(fold_scores)
        best_params_key = (
            fold_scores_df.groupby("params", as_index=False)["rmse"].mean().sort_values("rmse").iloc[0]["params"]
        )
        best_params = next(params for params in candidate_params if str(params) == best_params_key)

        best_cv_predictions = []
        best_cv_rows = []
        for fold_idx, (val_start, val_end) in enumerate(folds, start=1):
            train_df, val_df = _train_validation_split(model_df, val_start, val_end)
            _, preds = _fit_generic_model(train_df, val_df, config, builder, best_params, target_mode)
            scored = val_df.copy()
            scored["prediction"] = preds
            scored["fold"] = fold_idx
            best_cv_predictions.append(scored)
            metrics = regression_metrics(scored["cnt"].to_numpy(), scored["prediction"].to_numpy())
            metrics.update({"model": model_name, "fold": fold_idx, "params": str(best_params)})
            best_cv_rows.append(metrics)

        pre_holdout_df = model_df.loc[model_df["date"] < holdout_start].copy()
        estimator, holdout_preds = _fit_generic_model(pre_holdout_df, holdout_df, config, builder, best_params, target_mode)
        holdout_scored = holdout_df.copy()
        holdout_scored["prediction"] = holdout_preds
        cv_scored = pd.concat(best_cv_predictions, ignore_index=True)
        importance_df = _permutation_feature_importance(estimator, holdout_scored, config, model_name)
        results[model_name] = ModelResult(
            name=model_name,
            best_params=best_params,
            estimator=estimator,
            cv_predictions=cv_scored,
            cv_metrics=pd.DataFrame(best_cv_rows),
            holdout_predictions=holdout_scored,
            holdout_metrics=_evaluate_predictions(holdout_scored, model_name, "holdout"),
            feature_importance=importance_df,
        )
    return results


def select_best_model(results: Dict[str, ModelResult]) -> str:
    summary_rows = []
    for model_name, result in results.items():
        avg_rmse = result.cv_metrics["rmse"].mean()
        summary_rows.append({"model": model_name, "mean_cv_rmse": avg_rmse})
    return pd.DataFrame(summary_rows).sort_values("mean_cv_rmse").iloc[0]["model"]


def summarize_cv_metrics(results: Dict[str, ModelResult]) -> pd.DataFrame:
    rows = []
    for model_name, result in results.items():
        fold_df = result.cv_metrics.copy()
        fold_df["model"] = model_name
        rows.append(fold_df)
    return pd.concat(rows, ignore_index=True)


def summarize_holdout_predictions(results: Dict[str, ModelResult]) -> pd.DataFrame:
    frames = []
    for model_name, result in results.items():
        df = result.holdout_predictions.copy()
        df["model"] = model_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_feature_importance(results: Dict[str, ModelResult]) -> pd.DataFrame:
    frames = []
    for model_name, result in results.items():
        if result.feature_importance.empty:
            continue
        df = result.feature_importance.copy()
        df["model"] = model_name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std", "model"])
    return pd.concat(frames, ignore_index=True)
