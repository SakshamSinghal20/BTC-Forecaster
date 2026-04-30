from __future__ import annotations

import numpy as np

from src.backtest import run_backtest
from src.prediction import ForecastConfig


def test_backtest_uses_only_prior_prices() -> None:
    prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float)
    config = ForecastConfig(lookback=3, df=5.0, interval_scale=1.0, num_simulations=1000)

    predictions = run_backtest(prices, target_count=4, config=config, include_metadata=True)

    assert len(predictions) == 4
    assert [row["target_index"] for row in predictions] == [6, 7, 8, 9]
    for row in predictions:
        assert row["as_of_index"] == row["target_index"] - 1
        assert row["history_count"] == row["target_index"]
        assert row["actual"] == prices[row["target_index"]]


def test_conformal_backtest_preserves_required_output_shape() -> None:
    prices = 100 * np.exp(np.cumsum(np.sin(np.arange(180) / 8) * 0.002 + 0.0001))
    config = ForecastConfig(
        lookback=24,
        volatility_model="ensemble",
        conformal_enabled=True,
        conformal_min_history=10,
        num_simulations=1000,
    )

    predictions = run_backtest(prices, target_count=60, config=config)

    assert len(predictions) == 60
    assert set(predictions[0]) == {"actual", "lower", "upper"}
    assert all(row["upper"] >= row["lower"] for row in predictions)
