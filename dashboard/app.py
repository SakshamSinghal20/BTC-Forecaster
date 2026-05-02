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
from src.persistence import (
    load_history,
    persistence_stats,
    record_prediction,
    resolve_actuals,
)
from src.prediction import ForecastConfig, predict_price_range


METRICS_PATHS = (ROOT / "backtest_metrics.json", ROOT / "data" / "backtest_metrics.json")

# ═══════════════════════════════════════════════════════════════════
# Page config & custom CSS
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BTC Forecaster", layout="wide")

st.markdown(
    """
    <style>
    /* Kill the huge default top padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }

    /* Tighter metric spacing */
    [data-testid="stMetric"] {
        padding: 0.45rem 0;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
    }

    /* Section labels */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.15rem;
        padding-left: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════
# Chart builders
# ═══════════════════════════════════════════════════════════════════


def build_price_chart(df: pd.DataFrame, lower: float, upper: float) -> go.Figure:
    recent = df.tail(50).reset_index(drop=True).copy()
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
        height=400,
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
        height=220,
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title="Hourly sigma",
    )
    return fig


def build_history_chart(history: list[dict]) -> go.Figure | None:
    """Timeline chart showing past predictions with actuals."""
    resolved = [r for r in history if r.get("actual") is not None]
    pending = [r for r in history if r.get("actual") is None]

    if not resolved and not pending:
        return None

    fig = go.Figure()

    if resolved:
        r_times = [pd.Timestamp(r["target_time"]) for r in resolved]
        r_lowers = [float(r["lower"]) for r in resolved]
        r_uppers = [float(r["upper"]) for r in resolved]

        fig.add_trace(
            go.Scatter(
                x=r_times + r_times[::-1],
                y=r_uppers + r_lowers[::-1],
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.12)",
                line={"color": "rgba(37, 99, 235, 0)"},
                hoverinfo="skip",
                name="Predicted range",
            )
        )
        hits = [r for r in resolved if r.get("hit") is True]
        if hits:
            fig.add_trace(
                go.Scatter(
                    x=[pd.Timestamp(r["target_time"]) for r in hits],
                    y=[float(r["actual"]) for r in hits],
                    mode="markers",
                    marker={"color": "#16a34a", "size": 7, "symbol": "circle"},
                    name="Hit ✓",
                )
            )
        misses = [r for r in resolved if r.get("hit") is False]
        if misses:
            fig.add_trace(
                go.Scatter(
                    x=[pd.Timestamp(r["target_time"]) for r in misses],
                    y=[float(r["actual"]) for r in misses],
                    mode="markers",
                    marker={"color": "#dc2626", "size": 9, "symbol": "x"},
                    name="Miss ✗",
                )
            )

    if pending:
        p_times = [pd.Timestamp(r["target_time"]) for r in pending]
        p_lowers = [float(r["lower"]) for r in pending]
        p_uppers = [float(r["upper"]) for r in pending]
        fig.add_trace(
            go.Scatter(
                x=p_times + p_times[::-1],
                y=p_uppers + p_lowers[::-1],
                fill="toself",
                fillcolor="rgba(156, 163, 175, 0.18)",
                line={"color": "rgba(156, 163, 175, 0)"},
                hoverinfo="skip",
                name="Pending",
            )
        )

    fig.update_layout(
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        height=300,
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title="USDT",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def build_history_table(history: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reversed(history):
        actual = r.get("actual")
        hit = r.get("hit")
        if hit is True:
            status = "✅ Hit"
        elif hit is False:
            status = "❌ Miss"
        else:
            status = "⏳ Pending"
        rows.append(
            {
                "Target Time (UTC)": r.get("target_time", ""),
                "Price at Prediction": f"${float(r['current_price']):,.2f}",
                "Predicted Range": f"${float(r['lower']):,.2f} – ${float(r['upper']):,.2f}",
                "Width": f"${float(r['upper']) - float(r['lower']):,.2f}",
                "Actual": f"${actual:,.2f}" if actual is not None else "—",
                "Status": status,
                "Regime": str(r.get("regime", "")).title(),
            }
        )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Data loading & prediction
# ═══════════════════════════════════════════════════════════════════

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

# Persist this prediction
bar_close_time = df["close_time"].iloc[-1]
history = load_history()
history = record_prediction(
    history, prediction, bar_close_time, lower=lower, upper=upper,
)
history = resolve_actuals(history, df)

# ═══════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════

st.title("₿ BTC Forecaster")

# ── Row 1: Backtest + Price ──────────────────────────────────────

st.markdown('<p class="section-label">Backtest Performance</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Coverage", f"{metrics.get('coverage', 0.0) * 100:.2f}%")
c2.metric("Avg Width", money(float(metrics.get("mean_width", 0.0))))
c3.metric("Winkler Score", f"{float(metrics.get('mean_winkler', 0.0)):,.2f}")
c4.metric("Latest BTC Close", money(current_price))

st.divider()

# ── Row 2: Current prediction ────────────────────────────────────

st.markdown('<p class="section-label">Next-Hour 95 % Prediction</p>', unsafe_allow_html=True)
p1, p2, p3, p4 = st.columns(4)
p1.metric("Lower Bound", money(lower), pct(lower_pct))
p2.metric("Upper Bound", money(upper), pct(upper_pct))
p3.metric("Range Width", money(upper - lower))
p4.metric("Hourly σ", f"{float(prediction['volatility']) * 100:.3f}%")

m1, m2, m3 = st.columns(3)
m1.metric("Volatility Regime", str(prediction["regime"]).title())
m2.metric("Volatility Model", str(prediction["volatility_model"]).title())
m3.metric("Distribution", str(prediction["distribution"]).replace("_", " ").title())

# ── Charts ───────────────────────────────────────────────────────

tab_price, tab_vol = st.tabs(["📈 Price & Prediction", "📊 Realized Volatility"])

with tab_price:
    st.plotly_chart(build_price_chart(df, lower=lower, upper=upper), use_container_width=True)

with tab_vol:
    st.plotly_chart(build_volatility_chart(df), use_container_width=True)

# ── Prediction History (Part C) ──────────────────────────────────

st.divider()
st.markdown('<p class="section-label">Prediction History</p>', unsafe_allow_html=True)

stats = persistence_stats(history)

if stats["total"] == 0:
    st.info("No prediction history yet — predictions will accumulate as new bars close.")
else:
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Recorded", stats["total"])
    h2.metric("Resolved", stats["resolved"])
    h3.metric("Pending", stats["pending"])
    h4.metric(
        "Live Coverage",
        f"{stats['live_coverage'] * 100:.1f}%" if stats.get("live_coverage") is not None else "—",
    )
    h5.metric("Hits / Misses", f"{stats['hits']} / {stats['misses']}")

    history_fig = build_history_chart(history)
    if history_fig is not None:
        st.plotly_chart(history_fig, use_container_width=True)

    with st.expander("Full Prediction Log", expanded=False):
        st.dataframe(build_history_table(history), use_container_width=True, hide_index=True)

# ── Run Details ──────────────────────────────────────────────────

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
            "persistence_stats": stats,
        }
    )
