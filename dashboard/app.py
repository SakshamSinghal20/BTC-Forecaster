from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import estimate_conformal_adjustment
from src.data_fetch import fetch_btcusdt_bars
from src.features import latest_feature_snapshot
from src.prediction import ForecastConfig, predict_price_range


METRICS_PATHS = (ROOT / "backtest_metrics.json", ROOT / "data" / "backtest_metrics.json")


def money(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    return f"{value:+.2f}%"


@st.cache_data(ttl=300, show_spinner=False)
def load_live_data() -> pd.DataFrame:
    return fetch_btcusdt_bars(limit=500)


@st.cache_data(ttl=60, show_spinner=False)
def load_backtest_metrics() -> dict:
    for path in METRICS_PATHS:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def config_from_metrics(metrics: dict) -> ForecastConfig:
    if "config" not in metrics:
        return ForecastConfig()
    return ForecastConfig.from_mapping(metrics["config"])


def build_price_chart(df: pd.DataFrame, lower: float, upper: float) -> go.Figure:
    recent = df.tail(50).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(recent["close_time"]) + list(reversed(recent["close_time"])),
            y=[upper] * len(recent) + [lower] * len(recent),
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.14)",
            line={"color": "rgba(37, 99, 235, 0)"},
            hoverinfo="skip",
            name="95% range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent["close_time"],
            y=recent["close"],
            mode="lines",
            line={"color": "#111827", "width": 2},
            name="BTCUSDT close",
        )
    )
    fig.update_layout(
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        height=430,
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title="USDT",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def build_volatility_chart(df: pd.DataFrame) -> go.Figure:
    recent = df.tail(180).copy()
    prices = recent["close"].astype(float)
    log_returns = np.log(prices / prices.shift(1))
    rolling_vol = log_returns.rolling(24).std()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["close_time"],
            y=rolling_vol,
            mode="lines",
            line={"color": "#dc2626", "width": 2},
            name="24h realized volatility",
        )
    )
    fig.update_layout(
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        height=260,
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title="Hourly sigma",
    )
    return fig


st.set_page_config(page_title="BTC Forecaster", layout="wide")
st.title("BTC Forecaster")

metrics = load_backtest_metrics()
df = load_live_data()

if df.empty:
    st.error("No BTCUSDT bars were returned by Binance Vision.")
    st.stop()

config = config_from_metrics(metrics)
closes = df["close"].to_numpy(dtype=float)
prediction = predict_price_range(closes, config)
features = latest_feature_snapshot(df)

current_price = float(closes[-1])
lower = float(prediction["lower"])
upper = float(prediction["upper"])
conformal_adjustment = 0.0
if config.conformal_enabled:
    conformal_adjustment = estimate_conformal_adjustment(closes, config=config)
    lower -= conformal_adjustment
    upper += conformal_adjustment

lower_pct = (lower / current_price - 1) * 100
upper_pct = (upper / current_price - 1) * 100

metric_cols = st.columns(4)
metric_cols[0].metric("Backtest Coverage", f"{metrics.get('coverage', 0.0) * 100:.2f}%")
metric_cols[1].metric("Average Width", money(float(metrics.get("mean_width", 0.0))))
metric_cols[2].metric("Winkler Score", f"{float(metrics.get('mean_winkler', 0.0)):,.2f}")
metric_cols[3].metric("Latest Closed BTC", money(current_price))

range_cols = st.columns(3)
range_cols[0].metric("95% Lower Bound", money(lower), pct(lower_pct))
range_cols[1].metric("95% Upper Bound", money(upper), pct(upper_pct))
range_cols[2].metric("Range Width", money(upper - lower))

model_cols = st.columns(4)
model_cols[0].metric("Volatility Regime", str(prediction["regime"]).title())
model_cols[1].metric("Volatility Model", str(prediction["volatility_model"]).title())
model_cols[2].metric("Distribution", str(prediction["distribution"]).replace("_", " ").title())
model_cols[3].metric("Hourly Sigma", f"{float(prediction['volatility']) * 100:.3f}%")

st.plotly_chart(build_price_chart(df, lower=lower, upper=upper), use_container_width=True)
st.plotly_chart(build_volatility_chart(df), use_container_width=True)

with st.expander("Run Details", expanded=False):
    st.json(
        {
            "latest_closed_bar_utc": df["close_time"].iloc[-1].isoformat(),
            "model_config": config.to_dict(),
            "prediction_volatility": prediction["volatility"],
            "volatility_breakdown": {
                "rolling": prediction["rolling_volatility"],
                "ewma": prediction["ewma_volatility"],
                "garch": prediction["garch_volatility"],
            },
            "latest_features": features,
            "conformal_adjustment": conformal_adjustment,
            "conditional_coverage": metrics.get("conditional_coverage", {}),
            "backtest_generated_at_utc": metrics.get("generated_at_utc"),
        }
    )
