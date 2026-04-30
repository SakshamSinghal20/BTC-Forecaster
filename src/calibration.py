from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.backtest import calibrate_config
from src.prediction import ForecastConfig


def calibrate(
    data: pd.DataFrame,
    lookback_range: range | Iterable[int] = range(10, 51, 5),
    df_range: range | Iterable[int] = range(4, 8),
    scale_range: Iterable[float] = (0.9, 0.95, 1.0, 1.05, 1.1),
    volatility_models: Iterable[str] = ("rolling", "ewma", "garch", "ensemble"),
    distributions: Iterable[str] = ("student_t", "mixture"),
    target_coverage: float = 0.95,
) -> dict:
    """Grid search for a config near target coverage, then lowest Winkler score."""
    configs = [
        ForecastConfig(
            lookback=int(lookback),
            df=float(df_value),
            interval_scale=float(scale),
            num_simulations=10_000,
            confidence=0.95,
            seed=42,
            volatility_model=volatility_model,
            distribution=distribution,
            ewma_span=int(lookback),
        )
        for lookback in lookback_range
        for df_value in df_range
        for scale in scale_range
        for volatility_model in volatility_models
        for distribution in distributions
    ]
    best_config, best_metrics, predictions, rows = calibrate_config(
        data,
        target_count=720,
        target_coverage=target_coverage,
        configs=configs,
    )
    return {
        "best_config": best_config.to_dict(),
        "best_metrics": best_metrics,
        "predictions": predictions,
        "all_results": rows,
    }


def broad_calibration_grid() -> dict[str, list[float] | list[int]]:
    """Return the broader grid used by the production backtest script."""
    return {
        "lookback": [24, 36, 48, 72, 120, 168, 240],
        "df": [4, 5, 7, 10],
        "scale": [round(value, 2) for value in np.arange(0.70, 2.51, 0.05)],
        "volatility_model": ["rolling", "ewma", "garch", "ensemble"],
        "distribution": ["student_t", "mixture", "historical"],
    }
