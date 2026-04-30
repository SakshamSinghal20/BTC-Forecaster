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


def test_advanced_volatility_models_are_supported() -> None:
    prices = 100 * np.exp(np.cumsum(np.linspace(-0.002, 0.003, 220)))
    for model in ("rolling", "ewma", "garch", "ensemble"):
        config = ForecastConfig(
            lookback=36,
            volatility_model=model,
            distribution="mixture",
            num_simulations=1000,
        )
        prediction = predict_price_range(prices, config)

        assert prediction["upper"] > prediction["lower"] > 0
        assert prediction["volatility"] > 0
        assert prediction["regime"] in {"low", "medium", "high"}
