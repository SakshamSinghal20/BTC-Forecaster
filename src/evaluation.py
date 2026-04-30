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


def evaluate(predictions: Iterable[Prediction], alpha: float = 0.05) -> dict[str, float]:
    rows = list(predictions)
    return {
        "coverage": calculate_coverage(rows),
        "mean_width": calculate_mean_width(rows),
        "mean_winkler": calculate_winkler_score(rows, alpha=alpha),
    }

