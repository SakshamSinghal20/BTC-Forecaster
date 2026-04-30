from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import add_technical_indicators, latest_feature_snapshot


def test_technical_indicators_are_added() -> None:
    close = 100 + np.cumsum(np.sin(np.arange(80) / 5) + 0.2)
    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(10, 20, len(close)),
            "taker_buy_base": np.linspace(4, 12, len(close)),
        }
    )

    enriched = add_technical_indicators(df)
    snapshot = latest_feature_snapshot(df)

    assert "rsi_14" in enriched.columns
    assert "macd" in enriched.columns
    assert "atr_14" in enriched.columns
    assert "taker_buy_ratio" in snapshot
