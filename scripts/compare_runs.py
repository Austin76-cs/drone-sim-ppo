from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare experiment summaries")
    parser.add_argument("--root", type=str, default="checkpoints")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def sort_key(summary: dict[str, object]) -> tuple[float, ...]:
    metrics = summary["best_metrics"]
    assert isinstance(metrics, dict)
    return (
        float(metrics["success_rate"]),
        float(metrics["mean_gates_cleared"]),
        float(metrics["mean_completion"]),
        float(metrics["mean_reward"]),
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summaries: list[dict[str, object]] = []
    for path in root.glob("*/summary.json"):
        summary = json.loads(path.read_text())
        summary["summary_path"] = str(path)
        summaries.append(summary)

    summaries.sort(key=sort_key, reverse=True)
    print(
        f"{'Run':<28} {'Succ':>6} {'Gates':>7} {'Comp':>7} {'Reward':>9} {'FPS':>7} {'Cfg':<24}"
    )
    print("-" * 96)
    for summary in summaries[: args.limit]:
        metrics = summary["best_metrics"]
        training = summary.get("training", {})
        assert isinstance(metrics, dict)
        assert isinstance(training, dict)
        config_label = Path(str(summary["config"])).name
        print(
            f"{summary['run_name']:<28} "
            f"{float(metrics['success_rate']):>6.1%} "
            f"{float(metrics['mean_gates_cleared']):>7.2f} "
            f"{float(metrics['mean_completion']):>7.2f} "
            f"{float(metrics['mean_reward']):>9.2f} "
            f"{int(training.get('fps', 0)):>7d} "
            f"{config_label:<24}"
        )


if __name__ == "__main__":
    main()
