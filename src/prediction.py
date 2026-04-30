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
    volatility_model: str = "rolling"
    distribution: str = "student_t"
    ewma_span: int = 72
    garch_alpha: float = 0.08
    garch_beta: float = 0.90
    ensemble_rolling_weight: float = 0.35
    ensemble_ewma_weight: float = 0.35
    ensemble_garch_weight: float = 0.30
    use_regimes: bool = True
    regime_lookback: int = 240
    low_vol_percentile: float = 33.0
    high_vol_percentile: float = 67.0
    low_regime_df: float = 10.0
    high_regime_df: float = 4.0
    low_regime_scale: float = 0.92
    high_regime_scale: float = 1.12
    mixture_tail_weight: float = 0.15
    historical_lookback: int = 240
    conformal_enabled: bool = False
    conformal_alpha: float = 0.05
    conformal_min_history: int = 60

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ForecastConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        cleaned = {key: values[key] for key in values if key in allowed}
        if "scale" in values and "interval_scale" not in cleaned:
            cleaned["interval_scale"] = values["scale"]
        return cls(**cleaned)

    def to_dict(self) -> dict[str, float | int | str | bool]:
        output: dict[str, float | int | str | bool] = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.generic):
                value = value.item()
            output[key] = value
        return output


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


def rolling_volatility_from_returns(
    returns: np.ndarray,
    lookback: int,
    min_sigma: float = 1e-8,
) -> float:
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if len(returns) < lookback:
        raise ValueError(f"Need at least {lookback} returns; got {len(returns)}")
    sigma = float(np.std(returns[-lookback:], ddof=1))
    return max(sigma, float(min_sigma))


def ewma_volatility_from_returns(
    returns: np.ndarray,
    span: int,
    min_sigma: float = 1e-8,
) -> float:
    if span < 2:
        raise ValueError("span must be at least 2")
    if len(returns) < 2:
        raise ValueError("at least two returns are required")

    alpha = 2 / (span + 1)
    variance = float(np.var(returns[: min(len(returns), span)], ddof=1))
    for value in returns[-max(span * 3, span) :]:
        variance = alpha * float(value) ** 2 + (1 - alpha) * variance
    return max(float(np.sqrt(max(variance, 0.0))), float(min_sigma))


def garch_volatility_from_returns(
    returns: np.ndarray,
    lookback: int,
    alpha: float = 0.08,
    beta: float = 0.90,
    min_sigma: float = 1e-8,
) -> float:
    if len(returns) < max(lookback, 3):
        raise ValueError(f"Need at least {max(lookback, 3)} returns; got {len(returns)}")
    if alpha < 0 or beta < 0 or alpha + beta >= 1:
        raise ValueError("GARCH alpha and beta must be non-negative and sum to less than 1")

    recent = returns[-lookback:]
    long_run_var = float(np.var(recent, ddof=1))
    omega = max(long_run_var * (1 - alpha - beta), min_sigma**2)
    variance = max(long_run_var, min_sigma**2)
    for value in recent:
        variance = omega + alpha * float(value) ** 2 + beta * variance
    return max(float(np.sqrt(max(variance, 0.0))), float(min_sigma))


def detect_volatility_regime(
    returns: np.ndarray,
    lookback: int = 72,
    regime_lookback: int = 240,
    low_percentile: float = 33.0,
    high_percentile: float = 67.0,
    min_sigma: float = 1e-8,
) -> dict[str, float | str]:
    if len(returns) < max(lookback, 5):
        return {"regime": "medium", "recent_volatility": float(min_sigma)}

    window = min(lookback, len(returns))
    regime_returns = returns[-min(len(returns), max(regime_lookback, window + 1)) :]
    recent_vol = rolling_volatility_from_returns(returns, window, min_sigma=min_sigma)

    recent_intensity = float(np.mean(np.abs(returns[-window:])))
    historical_intensity = np.abs(regime_returns)
    if len(historical_intensity) < 3:
        return {"regime": "medium", "recent_volatility": recent_vol}

    low_cutoff, high_cutoff = np.percentile(
        historical_intensity,
        [low_percentile, high_percentile],
    )
    if recent_intensity <= low_cutoff:
        regime = "low"
    elif recent_intensity >= high_cutoff:
        regime = "high"
    else:
        regime = "medium"
    return {
        "regime": regime,
        "recent_volatility": float(recent_vol),
        "low_cutoff": float(low_cutoff),
        "high_cutoff": float(high_cutoff),
        "recent_intensity": float(recent_intensity),
    }


def estimate_model_volatility(
    prices: np.ndarray,
    config: ForecastConfig | Mapping[str, Any] | None = None,
) -> dict[str, float | str]:
    cfg = as_forecast_config(config)
    returns = calculate_log_returns(np.asarray(prices, dtype=float))
    rolling = rolling_volatility_from_returns(returns, cfg.lookback, min_sigma=cfg.min_sigma)
    ewma = ewma_volatility_from_returns(
        returns,
        span=max(2, int(cfg.ewma_span)),
        min_sigma=cfg.min_sigma,
    )
    garch = garch_volatility_from_returns(
        returns,
        lookback=cfg.lookback,
        alpha=float(cfg.garch_alpha),
        beta=float(cfg.garch_beta),
        min_sigma=cfg.min_sigma,
    )

    model = str(cfg.volatility_model).lower()
    if model == "rolling":
        sigma = rolling
    elif model == "ewma":
        sigma = ewma
    elif model == "garch":
        sigma = garch
    elif model == "ensemble":
        weights = np.array(
            [
                cfg.ensemble_rolling_weight,
                cfg.ensemble_ewma_weight,
                cfg.ensemble_garch_weight,
            ],
            dtype=float,
        )
        if np.any(weights < 0) or float(np.sum(weights)) <= 0:
            raise ValueError("ensemble volatility weights must be non-negative")
        weights = weights / np.sum(weights)
        sigma = float(np.dot(weights, np.array([rolling, ewma, garch], dtype=float)))
    else:
        raise ValueError(f"Unknown volatility_model: {cfg.volatility_model}")

    regime = detect_volatility_regime(
        returns,
        lookback=cfg.lookback,
        regime_lookback=cfg.regime_lookback,
        low_percentile=cfg.low_vol_percentile,
        high_percentile=cfg.high_vol_percentile,
        min_sigma=cfg.min_sigma,
    )
    return {
        "volatility": float(max(sigma, cfg.min_sigma)),
        "rolling_volatility": float(rolling),
        "ewma_volatility": float(ewma),
        "garch_volatility": float(garch),
        "regime": str(regime["regime"]),
        "regime_recent_volatility": float(regime["recent_volatility"]),
    }


def _unit_variance_student_t(df: float, size: int, seed: int) -> np.ndarray:
    if df <= 2:
        raise ValueError("Student-t degrees of freedom must be greater than 2")
    if size < 100:
        raise ValueError("num_simulations must be at least 100")

    rng = np.random.default_rng(seed)
    shocks = rng.standard_t(df, size=size)
    return shocks / np.sqrt(df / (df - 2))


def _regime_adjusted_df(config: ForecastConfig, regime: str) -> float:
    if not config.use_regimes:
        return float(config.df)
    if regime == "high":
        return float(config.high_regime_df)
    if regime == "low":
        return float(config.low_regime_df)
    return float(config.df)


def _regime_adjusted_scale(config: ForecastConfig, regime: str) -> float:
    if not config.use_regimes:
        return float(config.interval_scale)
    if regime == "high":
        return float(config.interval_scale) * float(config.high_regime_scale)
    if regime == "low":
        return float(config.interval_scale) * float(config.low_regime_scale)
    return float(config.interval_scale)


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


@lru_cache(maxsize=256)
def _normal_quantiles(num_simulations: int, confidence: float, seed: int) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(num_simulations)
    alpha = 1 - confidence
    lower, upper = np.quantile(shocks, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


@lru_cache(maxsize=256)
def _mixture_quantiles(
    df: float,
    tail_weight: float,
    num_simulations: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    if not 0 <= tail_weight <= 1:
        raise ValueError("mixture_tail_weight must be between 0 and 1")
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal(num_simulations)
    tail = rng.standard_t(df, size=num_simulations) / np.sqrt(df / (df - 2))
    use_tail = rng.random(num_simulations) < tail_weight
    shocks = np.where(use_tail, tail, normal)
    shocks = (shocks - float(np.mean(shocks))) / float(np.std(shocks, ddof=1))
    alpha = 1 - confidence
    lower, upper = np.quantile(shocks, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def _historical_quantiles(
    returns: np.ndarray,
    lookback: int,
    confidence: float,
    min_sigma: float,
) -> tuple[float, float]:
    history = returns[-min(len(returns), max(lookback, 30)) :]
    if len(history) < 30:
        return _shock_quantiles(5.0, 10_000, confidence, 42)
    centered = history - float(np.mean(history))
    sigma = max(float(np.std(centered, ddof=1)), float(min_sigma))
    shocks = centered / sigma
    alpha = 1 - confidence
    lower, upper = np.quantile(shocks, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def _distribution_quantiles(
    config: ForecastConfig,
    returns: np.ndarray,
    regime: str,
) -> tuple[float, float, float]:
    distribution = str(config.distribution).lower()
    adjusted_df = _regime_adjusted_df(config, regime)
    if distribution == "student_t":
        lower_q, upper_q = _shock_quantiles(
            adjusted_df,
            int(config.num_simulations),
            float(config.confidence),
            int(config.seed),
        )
    elif distribution == "normal":
        lower_q, upper_q = _normal_quantiles(
            int(config.num_simulations),
            float(config.confidence),
            int(config.seed),
        )
    elif distribution == "mixture":
        lower_q, upper_q = _mixture_quantiles(
            adjusted_df,
            float(config.mixture_tail_weight),
            int(config.num_simulations),
            float(config.confidence),
            int(config.seed),
        )
    elif distribution == "historical":
        lower_q, upper_q = _historical_quantiles(
            returns,
            lookback=int(config.historical_lookback),
            confidence=float(config.confidence),
            min_sigma=float(config.min_sigma),
        )
    else:
        raise ValueError(f"Unknown distribution: {config.distribution}")
    return lower_q, upper_q, adjusted_df


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
    returns = calculate_log_returns(price_array)
    volatility_details = estimate_model_volatility(price_array, cfg)
    sigma = float(volatility_details["volatility"])
    regime = str(volatility_details["regime"])

    lower_q, upper_q, adjusted_df = _distribution_quantiles(cfg, returns, regime)
    scaled_sigma = sigma * _regime_adjusted_scale(cfg, regime)
    log_center = -0.5 * scaled_sigma**2
    lower = current_price * np.exp(log_center + scaled_sigma * lower_q)
    upper = current_price * np.exp(log_center + scaled_sigma * upper_q)

    return {
        "lower": float(min(lower, upper)),
        "upper": float(max(lower, upper)),
        "current_price": current_price,
        "volatility": sigma,
        "rolling_volatility": float(volatility_details["rolling_volatility"]),
        "ewma_volatility": float(volatility_details["ewma_volatility"]),
        "garch_volatility": float(volatility_details["garch_volatility"]),
        "regime": regime,
        "distribution_df": float(adjusted_df),
        "volatility_model": str(cfg.volatility_model),
        "distribution": str(cfg.distribution),
    }
