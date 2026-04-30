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

