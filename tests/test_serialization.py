from __future__ import annotations

import json

from src.backtest import save_predictions_jsonl


def test_save_predictions_jsonl_keeps_required_keys(tmp_path) -> None:
    predictions = [
        {
            "actual": 100.0,
            "lower": 98.0,
            "upper": 102.0,
            "target_index": 7,
        }
    ]
    output = tmp_path / "backtest_results.jsonl"

    save_predictions_jsonl(predictions, output)

    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row == {"actual": 100.0, "lower": 98.0, "upper": 102.0}

