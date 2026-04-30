from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ForecastConfig:
    lookback: int = 72
    df: float = 5.0
    interval_scale: float = 1.0
    num_simulations: int = 10_000
    confidence: float = 0.95
    seed: int = 42
    min_sigma: float = 1e-8

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ForecastConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        cleaned = {key: values[key] for key in values if key in allowed}
        if "scale" in values and "interval_scale" not in cleaned:
            cleaned["interval_scale"] = values["scale"]
        return cls(**cleaned)

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def as_forecast_config(config: ForecastConfig | Mapping[str, Any] | None) -> ForecastConfig:
    if config is None:
        return ForecastConfig()
    if isinstance(config, ForecastConfig):
        return config
    return ForecastConfig.from_mapping(config)


def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError("prices must be a one-dimensional array")
    if len(prices) < 2:
        raise ValueError("at least two prices are required")
    if np.any(prices <= 0):
        raise ValueError("prices must be positive")
    return np.diff(np.log(prices))


def estimate_hourly_volatility(
    prices: np.ndarray,
    lookback: int,
    min_sigma: float = 1e-8,
) -> float:
    if lookback < 2:
        raise ValueError("lookback must be at least 2")

    returns = calculate_log_returns(prices)
    if len(returns) < lookback:
        raise ValueError(
            f"Need at least {lookback + 1} prices to estimate volatility; got {len(prices)}"
        )

    recent_returns = returns[-lookback:]
    sigma = float(np.std(recent_returns, ddof=1))
    return max(sigma, float(min_sigma))


def _unit_variance_student_t(df: float, size: int, seed: int) -> np.ndarray:
    if df <= 2:
        raise ValueError("Student-t degrees of freedom must be greater than 2")
    if size < 100:
        raise ValueError("num_simulations must be at least 100")

    rng = np.random.default_rng(seed)
    shocks = rng.standard_t(df, size=size)
    return shocks / np.sqrt(df / (df - 2))


@lru_cache(maxsize=256)
def _shock_quantiles(
    df: float,
    num_simulations: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    shocks = _unit_variance_student_t(df, num_simulations, seed)
    alpha = 1 - confidence
    lower, upper = np.quantile(shocks, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def simulate_gbm(
    current_price: float,
    volatility: float,
    drift: float = 0.0,
    num_simulations: int = 10_000,
    df: float = 5.0,
    interval_scale: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate one-hour GBM prices with Student-t shocks."""
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")

    shocks = _unit_variance_student_t(df, num_simulations, seed)
    scaled_sigma = max(float(volatility), 0.0) * float(interval_scale)
    log_returns = float(drift) - 0.5 * scaled_sigma**2 + scaled_sigma * shocks
    return float(current_price) * np.exp(log_returns)


def calculate_prediction_interval(
    simulated_prices: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    prices = np.asarray(simulated_prices, dtype=float)
    if prices.ndim != 1 or len(prices) == 0:
        raise ValueError("simulated_prices must be a non-empty one-dimensional array")

    alpha = 1 - confidence
    lower, upper = np.quantile(prices, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def predict_price_range(
    prices: np.ndarray,
    config: ForecastConfig | Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Predict the next one-hour BTCUSDT 95% interval from historical closes only."""
    cfg = as_forecast_config(config)
    price_array = np.asarray(prices, dtype=float)
    current_price = float(price_array[-1])
    sigma = estimate_hourly_volatility(
        price_array,
        lookback=cfg.lookback,
        min_sigma=cfg.min_sigma,
    )

    lower_q, upper_q = _shock_quantiles(
        float(cfg.df),
        int(cfg.num_simulations),
        float(cfg.confidence),
        int(cfg.seed),
    )
    scaled_sigma = sigma * float(cfg.interval_scale)
    log_center = -0.5 * scaled_sigma**2
    lower = current_price * np.exp(log_center + scaled_sigma * lower_q)
    upper = current_price * np.exp(log_center + scaled_sigma * upper_q)

    return {
        "lower": float(min(lower, upper)),
        "upper": float(max(lower, upper)),
        "current_price": current_price,
        "volatility": sigma,
    }
