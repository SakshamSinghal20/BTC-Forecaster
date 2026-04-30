from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import evaluate
from src.prediction import ForecastConfig, as_forecast_config, predict_price_range


DEFAULT_LOOKBACKS = (72, 120, 168, 240)
DEFAULT_DFS = (4.0, 5.0, 7.0)
DEFAULT_INTERVAL_SCALES = tuple(round(value, 2) for value in np.arange(0.85, 1.26, 0.05))
DEFAULT_MODEL_PAIRS = (
    ("rolling", "student_t"),
    ("ewma", "student_t"),
    ("garch", "student_t"),
    ("ensemble", "student_t"),
    ("ensemble", "mixture"),
    ("rolling", "historical"),
)


def extract_close_prices(data: pd.DataFrame | Iterable[float] | np.ndarray) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain a close column")
        prices = data["close"].to_numpy(dtype=float)
    else:
        prices = np.asarray(list(data), dtype=float)

    if prices.ndim != 1 or len(prices) == 0:
        raise ValueError("prices must be a non-empty one-dimensional sequence")
    if np.any(prices <= 0):
        raise ValueError("prices must be positive")
    return prices


def iter_backtest_windows(
    prices: Iterable[float] | np.ndarray,
    target_count: int,
    lookback: int,
) -> Iterable[tuple[int, np.ndarray, float]]:
    price_array = extract_close_prices(prices)
    if target_count < 1:
        raise ValueError("target_count must be positive")
    if len(price_array) <= target_count:
        raise ValueError("Need warmup bars before the target backtest window")

    target_start = len(price_array) - target_count
    if target_start < lookback + 1:
        raise ValueError(
            f"Need at least {lookback + 1} warmup prices before target window; got {target_start}"
        )

    for target_index in range(target_start, len(price_array)):
        historical = price_array[:target_index]
        actual = float(price_array[target_index])
        yield target_index, historical, actual


def run_backtest(
    data: pd.DataFrame | Iterable[float] | np.ndarray,
    target_count: int = 720,
    config: ForecastConfig | dict[str, Any] | None = None,
    include_metadata: bool = False,
) -> list[dict[str, Any]]:
    cfg = as_forecast_config(config)
    prices = extract_close_prices(data)

    predictions: list[dict[str, Any]] = []
    conformity_scores: list[float] = []
    for target_index, historical, actual in iter_backtest_windows(
        prices,
        target_count=target_count,
        lookback=cfg.lookback,
    ):
        interval = predict_price_range(historical, cfg)
        base_lower = float(interval["lower"])
        base_upper = float(interval["upper"])
        conformal_adjustment = 0.0
        if cfg.conformal_enabled and len(conformity_scores) >= cfg.conformal_min_history:
            conformal_adjustment = float(
                np.quantile(conformity_scores, 1 - float(cfg.conformal_alpha))
            )
        lower = base_lower - max(conformal_adjustment, 0.0)
        upper = base_upper + max(conformal_adjustment, 0.0)
        row: dict[str, Any] = {
            "actual": float(actual),
            "lower": float(lower),
            "upper": float(upper),
        }
        if include_metadata:
            row.update(
                {
                    "target_index": int(target_index),
                    "as_of_index": int(target_index - 1),
                    "history_count": int(len(historical)),
                    "volatility": float(interval["volatility"]),
                    "rolling_volatility": float(interval["rolling_volatility"]),
                    "ewma_volatility": float(interval["ewma_volatility"]),
                    "garch_volatility": float(interval["garch_volatility"]),
                    "regime": str(interval["regime"]),
                    "volatility_model": str(interval["volatility_model"]),
                    "distribution": str(interval["distribution"]),
                    "distribution_df": float(interval["distribution_df"]),
                    "conformal_adjustment": float(conformal_adjustment),
                }
            )
        predictions.append(row)
        conformity_scores.append(max(base_lower - float(actual), float(actual) - base_upper, 0.0))

    return predictions


def estimate_conformal_adjustment(
    data: pd.DataFrame | Iterable[float] | np.ndarray,
    config: ForecastConfig | dict[str, Any] | None = None,
    calibration_count: int = 120,
) -> float:
    """Estimate an online conformal additive adjustment from recent resolved bars."""
    cfg = as_forecast_config(config)
    prices = extract_close_prices(data)
    if len(prices) < cfg.lookback + cfg.conformal_min_history + 2:
        return 0.0

    target_count = min(calibration_count, len(prices) - cfg.lookback - 1)
    target_start = len(prices) - target_count
    scores: list[float] = []
    for target_index in range(target_start, len(prices)):
        historical = prices[:target_index]
        actual = float(prices[target_index])
        interval = predict_price_range(historical, cfg)
        scores.append(
            max(
                float(interval["lower"]) - actual,
                actual - float(interval["upper"]),
                0.0,
            )
        )
    if len(scores) < cfg.conformal_min_history:
        return 0.0
    return float(max(np.quantile(scores, 1 - float(cfg.conformal_alpha)), 0.0))


def candidate_configs(
    num_simulations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> list[ForecastConfig]:
    configs: list[ForecastConfig] = []
    for lookback in DEFAULT_LOOKBACKS:
        for df in DEFAULT_DFS:
            for interval_scale in DEFAULT_INTERVAL_SCALES:
                for volatility_model, distribution in DEFAULT_MODEL_PAIRS:
                    configs.append(
                        ForecastConfig(
                            lookback=lookback,
                            df=df,
                            interval_scale=interval_scale,
                            num_simulations=num_simulations,
                            confidence=confidence,
                            seed=seed,
                            volatility_model=volatility_model,
                            distribution=distribution,
                            ewma_span=lookback,
                            historical_lookback=max(lookback, 120),
                        )
                    )

    # A small conformal pass keeps the guarantee-style interval option available
    # without doubling the full search space.
    for lookback in (72, 168, 240):
        for df in (5.0, 7.0):
            for interval_scale in (0.9, 1.0, 1.1):
                configs.append(
                    ForecastConfig(
                        lookback=lookback,
                        df=df,
                        interval_scale=interval_scale,
                        num_simulations=num_simulations,
                        confidence=confidence,
                        seed=seed,
                        volatility_model="ensemble",
                        distribution="student_t",
                        ewma_span=lookback,
                        conformal_enabled=True,
                    )
                )
    return configs


def calibrate_config(
    data: pd.DataFrame | Iterable[float] | np.ndarray,
    target_count: int = 720,
    target_coverage: float = 0.95,
    configs: Iterable[ForecastConfig] | None = None,
) -> tuple[ForecastConfig, dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    prices = extract_close_prices(data)
    search_space = list(configs) if configs is not None else candidate_configs()
    if not search_space:
        raise ValueError("At least one candidate config is required")

    best: tuple[tuple[float, float, float], ForecastConfig, dict[str, float], list[dict[str, Any]]] | None = None
    calibration_rows: list[dict[str, Any]] = []

    for cfg in search_space:
        try:
            predictions = run_backtest(prices, target_count=target_count, config=cfg)
        except ValueError:
            continue

        metrics = evaluate(predictions)
        sort_key = (
            abs(float(metrics["coverage"]) - target_coverage),
            float(metrics["mean_winkler"]),
            float(metrics["mean_width"]),
        )
        calibration_rows.append(
            {
                **metrics,
                "coverage_error": sort_key[0],
                "config": cfg.to_dict(),
            }
        )

        if best is None or sort_key < best[0]:
            best = (sort_key, cfg, metrics, predictions)

    if best is None:
        raise ValueError("No candidate config had enough warmup data")

    _, best_config, best_metrics, best_predictions = best
    calibration_rows.sort(
        key=lambda row: (
            abs(float(row["coverage"]) - target_coverage),
            float(row["mean_winkler"]),
            float(row["mean_width"]),
        )
    )
    return best_config, best_metrics, best_predictions, calibration_rows


def save_predictions_jsonl(predictions: Iterable[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            row = {
                "actual": float(prediction["actual"]),
                "lower": float(prediction["lower"]),
                "upper": float(prediction["upper"]),
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
