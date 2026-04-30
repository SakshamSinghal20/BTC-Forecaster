# BTC Forecaster Methodology

## Forecasting Flow

The model predicts the next closed hourly BTCUSDT price using only information available before the target hour.

1. Binance Vision supplies closed hourly OHLCV bars.
2. Closing prices are converted to log returns.
3. Volatility is estimated with one of four explainable estimators:
   - rolling standard deviation
   - EWMA volatility
   - dependency-free GARCH(1,1) recursion
   - weighted ensemble of rolling, EWMA, and GARCH
4. The latest volatility regime is classified as low, medium, or high relative to recent rolling volatility.
5. Shock quantiles come from Student-t, normal, mixture, or historical standardized returns.
6. The model maps quantile log returns back into lower/upper BTCUSDT prices.
7. Backtests can apply online conformal widening from prior resolved misses only.

## Leakage Controls

For a target bar at index `i`, the model receives `prices[:i]` and the actual is `prices[i]`. Volatility, regimes, distribution quantiles that depend on history, and conformal scores all use data before the target.

## Calibration

Calibration searches a compact grid over lookback, volatility model, distribution, Student-t degrees of freedom, and interval scale. The selected model minimizes:

1. absolute distance from 95% coverage
2. mean Winkler score
3. mean interval width

## Deferred Extensions

The codebase now has clean seams for technical indicators, quantile models, and richer data sources. Heavy ML libraries and production services are intentionally not included in the scoring path because they increase deployment risk and can overfit a 720-hour evaluation window.

