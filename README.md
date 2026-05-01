# BTC Forecaster

## Overview

This project implements a robust Bitcoin (BTCUSDT) price forecasting system designed to predict 95% confidence intervals for the next hourly price. Developed for the AlphaI / Polaris Bitcoin Price Prediction challenge, it features a comprehensive backtesting framework and a live Streamlit dashboard for real-time predictions and performance monitoring.

The system is built to address key challenges in financial forecasting, including volatility clustering and fat-tailed distributions, ensuring both accuracy and tightness of prediction intervals.

## Key Features

-   **Data Acquisition:** Fetches historical BTCUSDT hourly bars from the Binance Vision API, ensuring reliable and up-to-date market data.
-   **Advanced Volatility Modeling:** Employs multiple volatility estimation techniques, including:
    -   **Rolling Volatility:** Standard deviation of log returns over a defined lookback window.
    -   **Exponentially Weighted Moving Average (EWMA):** Provides more weight to recent observations for adaptive volatility estimation.
    -   **Generalized Autoregressive Conditional Heteroskedasticity (GARCH(1,1)):** Models time-varying volatility based on past squared returns and past volatility.
    -   **Ensemble Volatility:** A weighted combination of the above models for a more robust estimate.
-   **Volatility Regime Detection:** Dynamically classifies market conditions into low, medium, or high volatility regimes, adjusting model parameters (e.g., Student-t degrees of freedom and interval scaling) accordingly to enhance prediction accuracy.
-   **Fat-Tailed Distribution Handling:** Utilizes Student-t, normal, mixture, and historical shock distributions to accurately capture the fat-tailed nature of cryptocurrency returns, which is crucial for realistic interval predictions.
-   **Geometric Brownian Motion (GBM) Simulation:** Employs Monte Carlo simulations based on GBM to generate a distribution of possible future prices, from which prediction intervals are derived.
-   **No-Peeking Backtesting:** A rigorous backtesting framework ensures that predictions for each historical bar are made using only data available *prior* to that bar, preventing lookahead bias and providing a realistic assessment of model performance over 720 hourly bars.
-   **Conformal Prediction (Optional):** Incorporates an online conformal prediction mechanism to dynamically adjust prediction intervals based on the model\'s recent performance, further improving coverage reliability.
-   **Automated Calibration:** Optimizes model hyperparameters (lookback periods, Student-t degrees of freedom, interval scaling, volatility models, and distributions) through a grid search, targeting 95% coverage while minimizing the Winkler score.
-   **Comprehensive Evaluation Metrics:** Tracks and reports key performance indicators including:
    -   **Coverage:** The percentage of actual prices falling within the predicted interval (target: 95%).
    -   **Mean Width:** The average width of the prediction intervals (narrower is better for a given coverage).
    -   **Winkler Score:** A combined metric that penalizes both interval width and misses, providing a holistic measure of forecast quality.
-   **Live Streamlit Dashboard:** A user-friendly web interface displaying:
    -   Current BTCUSDT price.
    -   Next-hour 95% prediction range.
    -   Percentage deviation of the prediction range from the current price.
    -   Interactive chart of recent price history with the prediction ribbon.
    -   Summary of backtest metrics (Coverage, Mean Width, Winkler Score).
    -   Detailed run information, including volatility breakdown, regime classification, and model configuration.

## Project Structure

```
btc-forecaster/
├── src/                    # Core forecasting, backtest, calibration, and evaluation logic
│   ├── data_fetch.py       # Handles data retrieval from Binance Vision API
│   ├── volatility.py       # Implements various volatility estimation models
│   ├── gbm_simulation.py   # Compatibility shim for GBM simulation
│   ├── prediction.py       # Main prediction interface, GBM, and interval calculation
│   ├── backtest.py         # Backtesting framework and calibration engine
│   ├── evaluation.py       # Metrics calculation (Coverage, Winkler Score, etc.)
│   ├── calibration.py      # High-level calibration interface
│   └── features.py         # Optional technical indicator generation for dashboard
├── dashboard/app.py        # Streamlit web application for live predictions
├── scripts/run_backtest.py # CLI script to execute the full backtesting pipeline
├── config/                 # Configuration files for model parameters
│   ├── default_config.py   # Loads default or calibrated model configurations
│   └── model_config.yaml   # Human-readable model configuration (if used)
├── data/                   # Stores generated backtest results and metrics
│   ├── backtest_results.jsonl # Detailed prediction outputs
│   └── backtest_metrics.json  # Summary of backtest performance metrics
├── tests/                  # Unit tests for all core modules
├── docs/METHODOLOGY.md     # Documentation on the forecasting methodology
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation (this file)
├── LICENSE                 # Project license
└── .gitignore              # Git ignore file
```

## Getting Started

### Prerequisites

-   Python 3.9+
-   `pip` (Python package installer)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/SakshamSinghal20/BTC-Forecaster.git
    cd BTC-Forecaster
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Backtest

Execute the backtest script to evaluate the model\'s historical performance and calibrate optimal parameters. This will generate `backtest_results.jsonl` and `backtest_metrics.json` in the `data/` directory.

```bash
python scripts/run_backtest.py
```

### Running the Live Dashboard

Launch the Streamlit dashboard to view live BTC price predictions and backtest metrics.

```bash
streamlit run dashboard/app.py
```

For deployment on Streamlit Community Cloud or similar platforms, set the application entry point to `dashboard/app.py`.

### Running Tests

To ensure the integrity and correctness of the codebase, run the unit tests:

```bash
pytest
```

## Methodology

The forecasting methodology is designed for transparency and robustness:

1.  **Log Return Calculation:** Historical hourly log returns are computed, strictly using data available before the target prediction bar to avoid lookahead bias.
2.  **Volatility Estimation:** The next-hour volatility is estimated using the configured volatility model (rolling, EWMA, GARCH, or ensemble).
3.  **Regime Classification:** The current volatility regime (low, medium, high) is identified relative to recent historical volatility.
4.  **Shock Distribution Sampling:** Prediction interval quantiles are drawn from the configured shock distribution (Student-t, normal, mixture, or historical), with parameters adjusted based on the detected volatility regime.
5.  **Price Bound Reconstruction:** Log-return quantiles are converted back to BTCUSDT price bounds.
6.  **Conformal Adjustment (Optional):** The prediction bounds can be optionally widened using online conformal scores derived from previously resolved predictions, enhancing the reliability of the 95% coverage guarantee.

This modular approach allows for clear understanding and potential future enhancements, while maintaining a lightweight and deployable solution.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
