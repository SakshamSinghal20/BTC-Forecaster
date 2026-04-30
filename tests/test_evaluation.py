from __future__ import annotations

from src.evaluation import evaluate


def test_evaluate_known_predictions() -> None:
    predictions = [
        {"actual": 10.0, "lower": 9.0, "upper": 11.0},
        {"actual": 8.0, "lower": 9.0, "upper": 11.0},
        {"actual": 12.0, "lower": 9.0, "upper": 11.0},
    ]

    metrics = evaluate(predictions, alpha=0.05)

    assert metrics["coverage"] == 1 / 3
    assert metrics["mean_width"] == 2.0
    assert metrics["mean_winkler"] == (2.0 + 42.0 + 42.0) / 3

