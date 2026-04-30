from __future__ import annotations

import numpy as np
from src.prediction import (
    calculate_log_returns,
    detect_volatility_regime,
    estimate_hourly_volatility,
    ewma_volatility_from_returns,
    garch_volatility_from_returns,
)


def calculate_volatility(prices: np.ndarray, window: int = 20) -> float:
    """Calculate hourly volatility from recent historical log returns."""
    return estimate_hourly_volatility(prices, lookback=window)


def get_adaptive_volatility(prices: np.ndarray, window: int = 20) -> float:
    """Alias for a rolling recent-volatility estimate."""
    return calculate_volatility(prices, window=window)


def get_ewma_volatility(prices: np.ndarray, span: int = 20) -> float:
    """Calculate exponentially weighted hourly volatility from log returns."""
    returns = calculate_log_returns(np.asarray(prices, dtype=float))
    return ewma_volatility_from_returns(returns, span=span)


def get_garch_volatility(
    prices: np.ndarray,
    window: int = 72,
    alpha: float = 0.08,
    beta: float = 0.90,
) -> float:
    """Estimate next-hour volatility with a dependency-free GARCH(1,1) recursion."""
    returns = calculate_log_returns(np.asarray(prices, dtype=float))
    return garch_volatility_from_returns(returns, lookback=window, alpha=alpha, beta=beta)


def get_volatility_regime(
    prices: np.ndarray,
    window: int = 72,
    regime_lookback: int = 240,
) -> dict[str, float | str]:
    """Classify current volatility as low, medium, or high versus recent history."""
    returns = calculate_log_returns(np.asarray(prices, dtype=float))
    return detect_volatility_regime(returns, lookback=window, regime_lookback=regime_lookback)
