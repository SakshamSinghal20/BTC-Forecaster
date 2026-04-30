# BTC Forecaster

One-hour BTCUSDT prediction intervals for the AlphaI / Polaris Bitcoin Price Prediction challenge.

The project fetches hourly BTCUSDT bars from Binance Vision, runs a no-peeking backtest over the latest 720 target bars, and serves a Streamlit dashboard with the current closed BTC price, next-hour 95% prediction range, recent chart, and backtest metrics.

## Model

- Uses hourly close-to-close log returns.
- Estimates volatility only from bars that are available before each target bar.
- Uses a GBM-style log-return forecast with Student-t shocks normalized to unit variance.
- Calibrates lookback, Student-t degrees of freedom, and interval scale to target 95% coverage.
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
data/                    Generated copies of backtest artifacts
tests/                   Unit tests
```
