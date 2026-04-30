# BTC Forecaster

One-hour BTCUSDT prediction intervals for the AlphaI / Polaris Bitcoin Price Prediction challenge.

The project fetches hourly BTCUSDT bars from Binance Vision, runs a no-peeking backtest over the latest 720 target bars, and serves a Streamlit dashboard with the current closed BTC price, next-hour 95% prediction range, recent chart, and backtest metrics.

## Model

- Uses hourly close-to-close log returns.
- Estimates volatility only from bars that are available before each target bar.
- Supports rolling, EWMA, GARCH(1,1), and weighted ensemble volatility estimates.
- Detects low/medium/high volatility regimes and adjusts tail thickness and scale by regime.
- Supports Student-t, normal, mixture, and historical shock distributions.
- Includes optional online conformal interval widening from previously resolved predictions.
- Calibrates lookback, tail degrees of freedom, volatility model, distribution, and interval scale to target 95% coverage.
- Keeps `drift = 0.0` for the one-hour forecast.

## Run Backtest

```powershell
python scripts/run_backtest.py
```

This writes:

- `backtest_results.jsonl` and `data/backtest_results.jsonl` with exactly `actual`, `lower`, and `upper` fields.
- `backtest_metrics.json` and `data/backtest_metrics.json` with coverage, average width, Winkler score, and chosen parameters.

## Run Dashboard

```powershell
streamlit run dashboard/app.py
```

For Streamlit Community Cloud, deploy this repository and set the app entrypoint to:

```text
dashboard/app.py
```

## Validate

```powershell
pytest
```

## Structure

```text
src/                    Core forecasting, backtest, calibration, and evaluation code
dashboard/app.py         Streamlit app
scripts/run_backtest.py  Backtest CLI
config/default_config.py Calibrated config loader
config/model_config.yaml Human-readable model configuration
data/                    Generated copies of backtest artifacts
tests/                   Unit tests
```

## Methodology

The forecast is intentionally explainable:

1. Compute historical hourly log returns without using the target bar.
2. Estimate next-hour volatility from the configured model.
3. Classify the current volatility regime relative to recent rolling volatility.
4. Draw interval quantiles from the configured shock distribution.
5. Convert log-return quantiles back to BTCUSDT price bounds.
6. Optionally widen bounds using online conformal scores from previous resolved bars.

Heavy ML models, cross-asset data, and API/Docker production layers are left as documented extensions so the competition submission stays lightweight and easy to deploy.
