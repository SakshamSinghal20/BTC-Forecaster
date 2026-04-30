from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


Prediction = dict[str, Any]


def _prediction_arrays(predictions: Iterable[Prediction]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(predictions)
    if not rows:
        raise ValueError("At least one prediction is required")

    actuals = np.array([float(row["actual"]) for row in rows], dtype=float)
    lowers = np.array([float(row["lower"]) for row in rows], dtype=float)
    uppers = np.array([float(row["upper"]) for row in rows], dtype=float)

    if np.any(lowers > uppers):
        raise ValueError("Prediction lower bound cannot exceed upper bound")

    return actuals, lowers, uppers


def calculate_coverage(predictions: Iterable[Prediction]) -> float:
    actuals, lowers, uppers = _prediction_arrays(predictions)
    hits = (lowers <= actuals) & (actuals <= uppers)
    return float(np.mean(hits))


def calculate_mean_width(predictions: Iterable[Prediction]) -> float:
    _, lowers, uppers = _prediction_arrays(predictions)
    return float(np.mean(uppers - lowers))


def calculate_winkler_score(predictions: Iterable[Prediction], alpha: float = 0.05) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    actuals, lowers, uppers = _prediction_arrays(predictions)
    widths = uppers - lowers
    below_penalty = (actuals < lowers) * (2 / alpha) * (lowers - actuals)
    above_penalty = (actuals > uppers) * (2 / alpha) * (actuals - uppers)
    return float(np.mean(widths + below_penalty + above_penalty))


def winkler_decomposition(predictions: Iterable[Prediction], alpha: float = 0.05) -> dict[str, float]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    actuals, lowers, uppers = _prediction_arrays(predictions)
    widths = uppers - lowers
    below_distance = np.maximum(lowers - actuals, 0.0)
    above_distance = np.maximum(actuals - uppers, 0.0)
    miss_distance = below_distance + above_distance
    penalty = (2 / alpha) * miss_distance
    return {
        "mean_width_component": float(np.mean(widths)),
        "mean_penalty_component": float(np.mean(penalty)),
        "below_rate": float(np.mean(actuals < lowers)),
        "above_rate": float(np.mean(actuals > uppers)),
        "mean_miss_distance": float(np.mean(miss_distance)),
    }


def evaluate_by_group(
    predictions: Iterable[Prediction],
    group_key: str = "regime",
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    rows = [row for row in predictions if group_key in row]
    grouped: dict[str, list[Prediction]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)
    return {
        key: {
            **evaluate(group_rows, alpha=alpha),
            "count": float(len(group_rows)),
        }
        for key, group_rows in sorted(grouped.items())
    }


def evaluate(predictions: Iterable[Prediction], alpha: float = 0.05) -> dict[str, float]:
    rows = list(predictions)
    decomposition = winkler_decomposition(rows, alpha=alpha)
    return {
        "coverage": calculate_coverage(rows),
        "mean_width": calculate_mean_width(rows),
        "mean_winkler": calculate_winkler_score(rows, alpha=alpha),
        "miss_rate": 1 - calculate_coverage(rows),
        **decomposition,
    }
