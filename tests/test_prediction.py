from __future__ import annotations

import numpy as np

from src.prediction import ForecastConfig, predict_price_range


def test_prediction_interval_is_ordered_and_positive() -> None:
    prices = 100 * np.exp(np.linspace(0, 0.05, 80))
    config = ForecastConfig(lookback=24, df=5.0, interval_scale=1.1, num_simulations=1000)

    prediction = predict_price_range(prices, config)

    assert prediction["lower"] > 0
    assert prediction["upper"] > prediction["lower"]
    assert prediction["lower"] < prediction["current_price"] < prediction["upper"]

