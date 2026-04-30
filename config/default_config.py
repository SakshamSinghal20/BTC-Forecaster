from __future__ import annotations

import json
from pathlib import Path

from src.prediction import ForecastConfig


ROOT = Path(__file__).resolve().parents[1]


def _metrics_paths() -> tuple[Path, Path]:
    return ROOT / "backtest_metrics.json", ROOT / "data" / "backtest_metrics.json"


def get_default_config() -> dict:
    """Return the calibrated config when available, otherwise a safe baseline."""
    for path in _metrics_paths():
        if path.exists():
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if "config" in metrics:
                return ForecastConfig.from_mapping(metrics["config"]).to_dict()
    return ForecastConfig(
        lookback=72,
        df=5.0,
        interval_scale=1.05,
        volatility_model="ensemble",
        distribution="student_t",
        ewma_span=72,
        use_regimes=True,
    ).to_dict()


def get_calibration_grid() -> dict[str, list[float] | list[int]]:
    return {
        "lookback": [36, 72, 120, 168, 240],
        "df": [4, 5, 6, 7],
        "scale": [0.9, 0.95, 1.0, 1.05, 1.1],
        "volatility_model": ["rolling", "ewma", "garch", "ensemble"],
        "distribution": ["student_t", "mixture", "historical"],
    }

