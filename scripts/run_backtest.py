from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import calibrate_config, save_predictions_jsonl
from src.data_fetch import BINANCE_VISION_KLINES_URL, fetch_btcusdt_bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BTCUSDT one-hour backtest.")
    parser.add_argument("--limit", type=int, default=1000, help="Bars to fetch from Binance Vision.")
    parser.add_argument("--target-count", type=int, default=720, help="Number of target bars to predict.")
    parser.add_argument("--results", default=ROOT / "backtest_results.jsonl", type=Path)
    parser.add_argument("--metrics", default=ROOT / "backtest_metrics.json", type=Path)
    parser.add_argument("--data-dir", default=ROOT / "data", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    df = fetch_btcusdt_bars(limit=args.limit, api_url=BINANCE_VISION_KLINES_URL)
    best_config, metrics, predictions, calibration_rows = calibrate_config(
        df,
        target_count=args.target_count,
    )

    args.data_dir.mkdir(exist_ok=True)

    save_predictions_jsonl(predictions, args.results)
    data_results = args.data_dir / "backtest_results.jsonl"
    save_predictions_jsonl(predictions, data_results)

    payload = {
        **metrics,
        "prediction_count": len(predictions),
        "rows_fetched": int(len(df)),
        "data_start_utc": df["open_time"].iloc[0].isoformat(),
        "data_end_utc": df["close_time"].iloc[-1].isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_url": BINANCE_VISION_KLINES_URL,
        "config": best_config.to_dict(),
        "top_calibration_rows": calibration_rows[:10],
    }
    metrics_text = json.dumps(payload, indent=2)
    args.metrics.write_text(metrics_text, encoding="utf-8")
    data_metrics = args.data_dir / "backtest_metrics.json"
    data_metrics.write_text(metrics_text, encoding="utf-8")

    print(f"Wrote {args.results}")
    print(f"Wrote {data_results}")
    print(f"Wrote {args.metrics}")
    print(f"Wrote {data_metrics}")
    print(
        "coverage={coverage:.4f} mean_width={mean_width:.2f} "
        "mean_winkler={mean_winkler:.2f}".format(**metrics)
    )
    print(f"config={best_config.to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
