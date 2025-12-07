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
    resolve_model_path,
    prepare_quant_layers,
)


# Default location for stored training checkpoints
DEFAULT_CHECKPOINTS_DIR = Path(
    # "/Users/joseph/Library/CloudStorage/GoogleDrive-johnstonljoseph@gmail.com/My Drive/a40/checkpoints"
    "/workspace/a40/checkpoints"
)


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


def parse_train_layers(raw: str) -> tuple[int, ...]:
    values = tuple(int(tok) for tok in raw.split(",") if tok.strip())
    if not values:
        raise ValueError("--train-layers requires at least one integer index")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a quantized checkpoint via lm_eval.")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINTS_DIR),
        help="Directory containing checkpoint files (step_<n>.pt).",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=1000,
        help="Training step whose checkpoint should be evaluated.",
    )
    parser.add_argument(
        "--train-layers",
        type=str,
        required=True,
        help="Comma-separated decoder layer indices that were quantized (e.g. 0,1,2).",
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
    args.train_layers = parse_train_layers(args.train_layers)
    checkpoints_dir = Path(args.checkpoints_dir).expanduser()
    args.checkpoint_path = checkpoints_dir / f"step_{args.checkpoint_step}.pt"
    return args


def load_teacher(args: argparse.Namespace):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    resolved = resolve_model_path(Config.base_path)
    return load_model(resolved, device, dtype)


def load_student(args: argparse.Namespace, teacher: torch.nn.Module):
    model = copy.deepcopy(teacher)
    print("  Hydrating...")
    prepare_quant_layers(
        model,
        args.train_layers,
        set_scales=False,
    )
    print("  Loading checkpoint...")
    load_checkpoint(str(args.checkpoint_path), model, map_location=args.device)
    return model


def main():
    args = parse_args()
    print("Loading teacher...")
    teacher = load_teacher(args)
    print("Loading student...")
    student = load_student(args, teacher)

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
