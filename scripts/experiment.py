from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from dronesim.training.eval_utils import evaluate_torch_checkpoint


UPDATE_RE = re.compile(r"update=(?P<update>\d+)\s+timesteps=(?P<timesteps>\d+)\s+fps=(?P<fps>\d+)")
STAGE_RE = re.compile(r"stage=(?P<stage>\d+)\s+mastery=(?P<mastery>[0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a structured torch PPO experiment")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--run-name", required=True, type=str)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=131_072)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--train-stage", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--policy-device", type=str, default="cuda")
    parser.add_argument("--sim-device", type=str, default="cuda")
    parser.add_argument("--eval-sim-device", type=str, default="cpu")
    parser.add_argument("--notes", type=str, default="")
    return parser.parse_args()


def model_score(metrics: dict[str, object]) -> tuple[float, ...]:
    return (
        float(metrics["success_rate"]),
        float(metrics["mean_gates_cleared"]),
        float(metrics["mean_completion"]),
        float(metrics["mean_reward"]),
    )


def load_eval_history(run_dir: Path) -> dict[str, list[float]]:
    history_path = run_dir / "evaluations.npz"
    if not history_path.exists():
        return {}
    data = np.load(history_path)
    return {key: data[key].astype(float).tolist() for key in data.files}


def run_training(args: argparse.Namespace) -> dict[str, float | int]:
    cmd = [
        sys.executable,
        "scripts/train_torch_ppo.py",
        "--config",
        args.config,
        "--run-name",
        args.run_name,
        "--total-timesteps",
        str(args.total_timesteps),
        "--device",
        args.policy_device,
        "--sim-device",
        args.sim_device,
    ]
    if args.resume:
        cmd.extend(["--resume", args.resume])
    if args.train_stage is not None:
        cmd.extend(["--force-stage", str(args.train_stage)])
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).resolve().parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_update: dict[str, float | int] = {}
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        if match := UPDATE_RE.search(line):
            last_update["update"] = int(match.group("update"))
            last_update["timesteps"] = int(match.group("timesteps"))
            last_update["fps"] = int(match.group("fps"))
        if match := STAGE_RE.search(line):
            last_update["stage"] = int(match.group("stage"))
            last_update["mastery"] = float(match.group("mastery"))
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"training failed with exit code {return_code}")
    return last_update


def evaluate_checkpoint(model_path: Path, args: argparse.Namespace) -> dict[str, object]:
    policy, summary = evaluate_torch_checkpoint(
        model_path,
        episodes=args.eval_episodes,
        stage=args.stage,
        deterministic=True,
        policy_device="cpu",
        sim_device=args.eval_sim_device,
        config_path=args.config,
    )
    metrics = summary.to_dict()
    metrics["global_step"] = policy.global_step
    metrics["update_idx"] = policy.update_idx
    return metrics


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    run_dir = root / "checkpoints" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    training_state = run_training(args)

    best_model = run_dir / "best_model.pt"
    final_model = run_dir / "final_model.pt"
    if not final_model.exists():
        raise FileNotFoundError(final_model)

    best_metrics = evaluate_checkpoint(best_model if best_model.exists() else final_model, args)
    final_metrics = evaluate_checkpoint(final_model, args)
    history = load_eval_history(run_dir)

    summary = {
        "run_name": args.run_name,
        "config": args.config,
        "notes": args.notes,
        "total_timesteps": args.total_timesteps,
        "stage": args.stage,
        "policy_device": args.policy_device,
        "sim_device": args.sim_device,
        "eval_sim_device": args.eval_sim_device,
        "training": training_state,
        "best_model_path": str(best_model if best_model.exists() else final_model),
        "final_model_path": str(final_model),
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "best_score": model_score(best_metrics),
        "final_score": model_score(final_metrics),
        "eval_history": history,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nExperiment summary saved to {summary_path}")
    print(
        "best "
        f"success={best_metrics['success_rate']:.1%} "
        f"gates={best_metrics['mean_gates_cleared']:.2f} "
        f"completion={best_metrics['mean_completion']:.2f} "
        f"reward={best_metrics['mean_reward']:.2f}"
    )


if __name__ == "__main__":
    main()
