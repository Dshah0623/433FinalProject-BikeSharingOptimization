from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def underprediction_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred < y_true))


def overprediction_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred > y_true))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "mean_bias_error": mean_bias_error(y_true, y_pred),
        "underprediction_rate": underprediction_rate(y_true, y_pred),
        "overprediction_rate": overprediction_rate(y_true, y_pred),
    }


def segmented_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    segment_col: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    rows = []
    for segment_value, group in df.groupby(segment_col):
        metrics = regression_metrics(group[y_true_col].to_numpy(), group[y_pred_col].to_numpy())
        metrics[segment_col] = segment_value
        if prefix:
            metrics["segment_type"] = prefix
        rows.append(metrics)
    return pd.DataFrame(rows)
