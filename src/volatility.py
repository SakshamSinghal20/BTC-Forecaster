from __future__ import annotations

import numpy as np
import pandas as pd

from src.prediction import calculate_log_returns, estimate_hourly_volatility


def calculate_volatility(prices: np.ndarray, window: int = 20) -> float:
    """Calculate hourly volatility from recent historical log returns."""
    return estimate_hourly_volatility(prices, lookback=window)


def get_adaptive_volatility(prices: np.ndarray, window: int = 20) -> float:
    """Alias for a rolling recent-volatility estimate."""
    return calculate_volatility(prices, window=window)


def get_ewma_volatility(prices: np.ndarray, span: int = 20) -> float:
    """Calculate exponentially weighted hourly volatility from log returns."""
    if span < 2:
        raise ValueError("span must be at least 2")
    returns = calculate_log_returns(np.asarray(prices, dtype=float))
    if len(returns) < 2:
        raise ValueError("at least three prices are required")
    volatility = pd.Series(returns).ewm(span=span, adjust=False).std().iloc[-1]
    return float(max(volatility, 1e-8))

