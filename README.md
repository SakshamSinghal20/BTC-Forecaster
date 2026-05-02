# ₿ BTC Forecaster

**Live Dashboard →** [btc-forecasterr.streamlit.app](https://btc-forecasterr.streamlit.app/)

A Bitcoin (BTCUSDT) price forecasting system that predicts 95% confidence intervals for the next hourly price. Built for the **AlphaI × Polaris Bitcoin Price Prediction Challenge**.

## Backtest Results

| Metric | Value |
|---|---|
| **Coverage** | 95.00% (target: 95%) |
| **Mean Width** | $1,227.99 |
| **Winkler Score** | 1,698.27 |
| **Predictions** | 720 hourly bars |
| **Best Config** | GARCH(1,1) + Student-t (df=7) + regime detection |

## Key Features

-   **Data Acquisition:** Fetches historical BTCUSDT hourly bars from the Binance Vision API (`data-api.binance.vision`), the geo-unblocked endpoint that works in India without any API key.
-   **Advanced Volatility Modeling:** Employs multiple volatility estimation techniques:
    -   **Rolling Volatility:** Standard deviation of log returns over a defined lookback window.
    -   **EWMA:** Exponentially weighted moving average — recent bars carry more weight.
    -   **GARCH(1,1):** Models time-varying volatility using past squared returns and past variance.
    -   **Ensemble:** Weighted combination of all three (0.35 rolling, 0.35 EWMA, 0.30 GARCH).
-   **Volatility Regime Detection:** Classifies the current market into low, medium, or high volatility regimes by comparing recent return intensity against historical percentiles. The regime dynamically adjusts:
    -   Student-t degrees of freedom: low → df=10 (lighter tails), high → df=4 (heavier tails).
    -   Interval scaling: low → 0.92× (tighter), high → 1.12× (wider).
-   **Fat-Tailed Distributions:** Supports Student-t, normal, mixture, and historical shock distributions. The calibrated model uses Student-t to capture the fat-tailed nature of cryptocurrency returns.
-   **GBM Simulation:** Geometric Brownian Motion with drift correction (`-0.5σ²`) to produce martingale-consistent prediction intervals.
-   **No-Peeking Backtesting:** For each of the 720 target bars, predictions use only `prices[:target_index]` — the model never sees future data. This prevents lookahead bias.
-   **Automated Calibration:** Grid search over ~666 config combinations (lookback × df × scale × volatility model × distribution), selecting the config closest to 95% coverage with the lowest Winkler score.
-   **Conformal Prediction (Optional):** An online mechanism that widens intervals based on the model's recent prediction errors, available as a toggle.
-   **Prediction Persistence (Part C):** Every dashboard visit records the current prediction. On subsequent visits, past predictions are resolved against actual prices, building a growing timeline with live coverage tracking.

## Live Dashboard

The dashboard is deployed at **[btc-forecasterr.streamlit.app](https://btc-forecasterr.streamlit.app/)** and displays:

-   Current BTCUSDT price and next-hour 95% prediction range.
-   Interactive chart of the last 50 bars with the predicted range as a shaded ribbon.
-   Backtest metrics (Coverage, Mean Width, Winkler Score) as headline numbers.
-   Volatility regime, model configuration, and hourly sigma.
-   Realized volatility chart (24h rolling window).
-   **Prediction History:** A growing timeline of saved predictions showing hits (✅), misses (❌), and pending (⏳) outcomes with live coverage stats.

## Project Structure

```
BTC-Forecaster/
├── src/                        # Core forecasting logic
│   ├── data_fetch.py           # Binance Vision API data retrieval
│   ├── prediction.py           # GBM prediction, volatility models, regime detection
│   ├── backtest.py             # No-peeking backtesting + calibration engine
│   ├── evaluation.py           # Coverage, Mean Width, Winkler Score metrics
│   ├── calibration.py          # High-level calibration interface
│   ├── persistence.py          # Part C: prediction history storage & resolution
│   ├── volatility.py           # Volatility model convenience wrappers
│   ├── features.py             # Technical indicators (RSI, MACD, Bollinger, ATR)
│   └── gbm_simulation.py       # GBM compatibility shim
├── dashboard/
│   └── app.py                  # Streamlit live dashboard
├── scripts/
│   └── run_backtest.py         # CLI: run full backtest + calibration pipeline
├── config/
│   ├── default_config.py       # Loads calibrated or fallback config
│   └── model_config.yaml       # Human-readable config reference
├── data/
│   ├── backtest_results.jsonl   # 720 predictions (one per line)
│   ├── backtest_metrics.json    # Performance metrics + winning config
│   └── prediction_history.json  # Part C: persisted prediction log
├── tests/                      # Unit tests (prediction, backtest, evaluation, etc.)
├── docs/METHODOLOGY.md         # Forecasting methodology documentation
├── .streamlit/config.toml      # Streamlit theme + deployment config
├── requirements.txt            # Dependencies (numpy, pandas, streamlit, plotly, etc.)
├── LICENSE                     # MIT License
└── README.md
```

## Getting Started

### Prerequisites

-   Python 3.9+
-   `pip`

### Installation

```bash
git clone https://github.com/SakshamSinghal20/BTC-Forecaster.git
cd BTC-Forecaster
pip install -r requirements.txt
```

### Running the Backtest

Runs the full calibration grid search and generates `backtest_results.jsonl` + `backtest_metrics.json`:

```bash
python scripts/run_backtest.py
```

### Running the Dashboard Locally

```bash
streamlit run dashboard/app.py
```

### Running Tests

```bash
pytest
```

## Methodology

1.  **Log Return Calculation:** `r_t = ln(P_t / P_{t-1})` — computed strictly from data available before the target bar to prevent lookahead bias.
2.  **Volatility Estimation:** The next-hour σ is estimated using the calibrated model (GARCH(1,1) with lookback=168).
3.  **Regime Classification:** Recent return intensity is compared against the 33rd/67th percentiles of the last 240 bars to classify the regime as low, medium, or high.
4.  **Shock Quantiles:** 10,000 Student-t samples (regime-adjusted df) are drawn and normalized to unit variance. The 2.5th and 97.5th percentiles define the shock quantiles.
5.  **Price Bound Reconstruction:** `bound = P_current × exp(-0.5σ² + σ × quantile)` — the GBM formula converting log-return quantiles back to USDT price bounds.
6.  **Conformal Adjustment (Optional):** Intervals can be widened by the 95th percentile of recent conformity scores for guaranteed coverage.

## Bugs Spotted in Starter Notebook

-   The starter Colab uses daily USD/CHF data — it needed to be adapted for hourly BTCUSDT bars from Binance Vision.
-   The original GBM simulation did not include regime-based parameter adjustment, which is critical for Bitcoin's volatility clustering.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
