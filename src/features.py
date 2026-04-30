from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add explainable price/volume features without external TA dependencies."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    features = df.copy()
    close = features["close"].astype(float)
    high = features["high"].astype(float)
    low = features["low"].astype(float)
    volume = features["volume"].astype(float)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    features["macd"] = ema_12 - ema_26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    features["bollinger_mid_20"] = mid
    features["bollinger_upper_20"] = mid + 2 * std
    features["bollinger_lower_20"] = mid - 2 * std
    features["bollinger_width_20"] = (features["bollinger_upper_20"] - features["bollinger_lower_20"]) / close

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    features["atr_14"] = true_range.rolling(14).mean()
    features["volume_z_24"] = (volume - volume.rolling(24).mean()) / volume.rolling(24).std()

    if "taker_buy_base" in features.columns:
        taker_buy_base = features["taker_buy_base"].astype(float)
        features["taker_buy_ratio"] = taker_buy_base / volume.replace(0, np.nan)

    return features


def latest_feature_snapshot(df: pd.DataFrame) -> dict[str, float]:
    """Return the latest finite feature values for dashboard/debug display."""
    enriched = add_technical_indicators(df)
    latest = enriched.iloc[-1]
    fields = [
        "rsi_14",
        "macd",
        "macd_signal",
        "bollinger_width_20",
        "atr_14",
        "volume_z_24",
        "taker_buy_ratio",
    ]
    snapshot: dict[str, float] = {}
    for field in fields:
        if field in latest and pd.notna(latest[field]):
            snapshot[field] = float(latest[field])
    return snapshot

