"""Prediction history persistence for the BTC Forecaster dashboard.

Records one prediction per closed bar, resolves actuals when the
target bar closes, and computes live performance statistics.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = Path(__file__).resolve().parents[1] / "data" / "prediction_history.json"


def load_history(path: Path = DEFAULT_HISTORY_PATH) -> list[dict[str, Any]]:
    """Load the prediction history from a JSON file.

    Returns an empty list when the file does not exist, is empty, or
    contains malformed JSON — the dashboard should never crash because
    of a corrupt history file.
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        data = json.loads(text)
        if not isinstance(data, list):
            LOGGER.warning("prediction history is not a list — resetting")
            return []
        return data
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("could not read prediction history: %s", exc)
        return []


def _save_history(
    history: list[dict[str, Any]],
    path: Path = DEFAULT_HISTORY_PATH,
) -> None:
    """Write the full history list to disk atomically.

    Writes to a temporary file first, then replaces the target — this
    prevents corruption if the process is interrupted mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(
            json.dumps(history, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        LOGGER.error("failed to save prediction history: %s", exc)
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _bar_close_key(bar_close_time: pd.Timestamp) -> str:
    """Normalise a bar close timestamp to a stable string key.

    All Binance 1-hour bars end at ``HH:59:59.999``.  We round down to
    the enclosing second so that minor millisecond drift never produces
    duplicate keys.
    """
    return bar_close_time.floor("s").isoformat()


def has_prediction_for_bar(
    history: list[dict[str, Any]],
    bar_close_time: pd.Timestamp,
) -> bool:
    """Return *True* if a prediction already exists for this bar close."""
    key = _bar_close_key(bar_close_time)
    return any(r.get("as_of_bar_close") == key for r in history)


def record_prediction(
    history: list[dict[str, Any]],
    prediction: dict[str, Any],
    bar_close_time: pd.Timestamp,
    lower: float,
    upper: float,
    path: Path = DEFAULT_HISTORY_PATH,
) -> list[dict[str, Any]]:
    """Append a new prediction to history if one does not already exist
    for this bar, then persist to disk.

    Parameters
    ----------
    history:
        The current in-memory history list (will be mutated).
    prediction:
        The dict returned by ``predict_price_range``.
    bar_close_time:
        ``close_time`` of the last closed bar used for the prediction.
    lower, upper:
        Final bounds **after** any conformal adjustment.
    path:
        File path for the JSON history.

    Returns the (possibly appended) history list.
    """
    if has_prediction_for_bar(history, bar_close_time):
        return history

    target_time = bar_close_time + pd.Timedelta(hours=1)

    record: dict[str, Any] = {
        "predicted_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "as_of_bar_close": _bar_close_key(bar_close_time),
        "target_time": _bar_close_key(target_time),
        "current_price": float(prediction["current_price"]),
        "lower": float(lower),
        "upper": float(upper),
        "volatility": float(prediction["volatility"]),
        "regime": str(prediction["regime"]),
        "actual": None,
        "hit": None,
    }
    history.append(record)
    _save_history(history, path)
    return history


def resolve_actuals(
    history: list[dict[str, Any]],
    bars_df: pd.DataFrame,
    path: Path = DEFAULT_HISTORY_PATH,
) -> list[dict[str, Any]]:
    """Fill in ``actual`` and ``hit`` for every prediction whose target
    bar has already closed.

    Matches each prediction's ``target_time`` against the bar DataFrame's
    ``close_time`` column (floored to the second for consistency).
    """
    if bars_df.empty or not history:
        return history

    # Build a lookup: floored close_time ISO → close price
    price_lookup: dict[str, float] = {}
    for _, row in bars_df.iterrows():
        ct = pd.Timestamp(row["close_time"])
        price_lookup[ct.floor("s").isoformat()] = float(row["close"])

    changed = False
    for record in history:
        if record.get("actual") is not None:
            continue  # already resolved

        target_key = record.get("target_time")
        if target_key and target_key in price_lookup:
            actual = price_lookup[target_key]
            record["actual"] = actual
            record["hit"] = record["lower"] <= actual <= record["upper"]
            changed = True

    if changed:
        _save_history(history, path)

    return history


def persistence_stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from the prediction history."""
    resolved = [r for r in history if r.get("actual") is not None]
    pending = [r for r in history if r.get("actual") is None]
    hits = [r for r in resolved if r.get("hit") is True]
    misses = [r for r in resolved if r.get("hit") is False]

    stats: dict[str, Any] = {
        "total": len(history),
        "resolved": len(resolved),
        "pending": len(pending),
        "hits": len(hits),
        "misses": len(misses),
    }

    if resolved:
        stats["live_coverage"] = len(hits) / len(resolved)
        widths = [float(r["upper"]) - float(r["lower"]) for r in resolved]
        stats["avg_width"] = sum(widths) / len(widths)
        # Live Winkler score
        alpha = 0.05
        winkler_total = 0.0
        for r in resolved:
            w = float(r["upper"]) - float(r["lower"])
            a = float(r["actual"])
            if a < float(r["lower"]):
                w += (2 / alpha) * (float(r["lower"]) - a)
            elif a > float(r["upper"]):
                w += (2 / alpha) * (a - float(r["upper"]))
            winkler_total += w
        stats["live_winkler"] = winkler_total / len(resolved)

    return stats
