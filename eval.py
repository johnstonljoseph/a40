import argparse
import copy
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM

from .main import (
    Config,
    load_checkpoint,
    load_model,
    load_custom_model,
    resolve_model_path,
    prepare_quant_layers,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a quantized checkpoint via lm_eval.")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="/workspace/a40/checkpoints/student_final",
        help="Directory containing checkpoint files (step_<n>.pt).",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=1000,
        help="Training step whose checkpoint should be evaluated.",
    )
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seq-len", type=int, default=Config.seq_len)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--dtype", type=str, default=Config.dtype)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["gsm8k"],
        help="Whitespace-separated lm_eval task names.",
    )
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--limit", type=float, default=0.05, help="Optional fraction per task (e.g., 0.01).")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()
    args.base_path = "/workspace/.hf_home/hub/models--allenai--Olmo-3-7B-Think"
    checkpoints_dir = Path(args.checkpoints_dir).expanduser()
    args.checkpoint_path = checkpoints_dir / f"checkpoint_{args.checkpoint_step}"
    return args


def extract_basic_metrics(results: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Flatten all numeric task metrics into a single dict for easy printing/logging."""
    metrics: dict[str, float] = {}
    for task, task_results in results.items():
        for metric_name, value in task_results.items():
            if isinstance(value, (int, float)):
                metrics[f"{task}/{metric_name}"] = float(value)
    return metrics


def print_results(results: Mapping[str, dict]) -> None:
    print(json.dumps(results.get("results", {}), indent=2))

    aggregate = results.get("aggregate")
    if aggregate is not None:
        print("\nAggregated:")
        print(json.dumps(aggregate, indent=2))
    else:
        print("\nAggregated metrics not provided for this run.")


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    print("Loading teacher...")
    teacher = load_model(args.base_path, device, dtype)
    print("Loading student...")
    student = load_custom_model(args.checkpoint_path, device, dtype)

    with torch.no_grad():
        print("Evaluating teacher...")
        teacher_results = lm_eval.simple_evaluate(
            model=HFLM(pretrained=teacher, batch_size=args.batch_size),
            model_args=None,
            tasks=args.tasks,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )
        print("Evaluating student...")
        student_results = lm_eval.simple_evaluate(
            model=HFLM(pretrained=student, batch_size=args.batch_size),
            model_args=None,
            tasks=args.tasks,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )

    print("\n== student ==")
    print_results(student_results)
    print("\n== teacher ==")
    print_results(teacher_results)

    def _json_default(obj):
        try:
            return obj.item()
        except AttributeError:
            return str(obj)

    payload = {
        "student": student_results,
        "teacher": teacher_results,
    }
    print("\nSaving results to", args.output)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


if __name__ == "__main__":
    main()
