from __future__ import annotations

from datetime import datetime, timezone
from time import sleep
from typing import Final

import pandas as pd
import requests


BINANCE_VISION_KLINES_URL: Final[str] = "https://data-api.binance.vision/api/v3/klines"


def fetch_btcusdt_bars(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 1000,
    api_url: str = BINANCE_VISION_KLINES_URL,
    drop_unclosed: bool = True,
    timeout: int = 30,
    retries: int = 2,
) -> pd.DataFrame:
    """Fetch BTCUSDT klines and return a typed DataFrame of closed bars."""
    if limit < 1 or limit > 1000:
        raise ValueError("Binance kline limit must be between 1 and 1000")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                api_url,
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=timeout,
            )
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                raise
            sleep(0.8 * (attempt + 1))
    else:
        raise RuntimeError("Failed to fetch Binance Vision data") from last_error

    raw_rows = response.json()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(raw_rows, columns=columns)
    if df.empty:
        return df

    df["open_time_ms"] = pd.to_numeric(df["open_time"], errors="raise").astype("int64")
    df["close_time_ms"] = pd.to_numeric(df["close_time"], errors="raise").astype("int64")
    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["trades"] = pd.to_numeric(df["trades"], errors="raise").astype("int64")

    if drop_unclosed:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        df = df.loc[df["close_time_ms"] <= now_ms].copy()

    return df.reset_index(drop=True)
