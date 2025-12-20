import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

from .utils import resolve_model_path
from . import custom_model  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a quantized checkpoint via lm_eval.")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="/workspace/a40/checkpoints",
        help="Directory containing checkpoint_<n> folders (or trainer_state.pt file inside one).",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        help="Training step whose checkpoint should be evaluated.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--include-teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate the teacher model in lm_eval mode (slower). Use --no-include-teacher to skip.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["gsm8k"],
        help="Whitespace-separated lm_eval task names.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=1.0,
        help="Number or fraction of examples per task (e.g., 1 or 0.01). Use <=0 for no cap.",
    )
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()
    args.base_path = resolve_model_path("/workspace/.hf_home/hub/models--allenai--Olmo-3-7B-Think")
    checkpoints_dir = Path(args.checkpoints_dir).expanduser()
    candidate = checkpoints_dir / str(args.checkpoint_name)
    if (candidate / "config.json").exists():
        args.checkpoint_path = str(candidate)
    else:
        raise FileNotFoundError(
            f"Could not find checkpoint directory {candidate} for step {args.checkpoint_name}. Expected config.json to exist."
        )
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

    tokenizer = AutoTokenizer.from_pretrained(args.base_path)

    teacher_results = None
    with torch.no_grad():
        print("Evaluating student...")

        # Interpret non-positive limit as "no cap" for lm_eval.
        limit = None if args.limit is not None and args.limit <= 0 else args.limit

        student_lm = HFLM(
            pretrained=args.checkpoint_path,
            batch_size=args.batch_size,
            device=args.device,
            tokenizer=tokenizer,
            max_length=int(args.seq_len),
        )
        # Avoid HF warning about both max_new_tokens and max_length by clearing
        # the model's generation_config max_new_tokens; we rely on max_gen_toks
        # (set below) to cap generated tokens.
        student_lm.model.generation_config.max_new_tokens = None

        student_results = lm_eval.simple_evaluate(
            model=student_lm,
            tasks=args.tasks,
            batch_size=args.batch_size,
            limit=limit,
            confirm_run_unsafe_code=True,
        )

        if args.include_teacher:
            print("Evaluating teacher...")
            teacher_lm = HFLM(
                pretrained=args.base_path,
                batch_size=args.batch_size,
                device=args.device,
                tokenizer=tokenizer,
                max_length=int(args.seq_len),
            )
            teacher_lm.model.generation_config.max_new_tokens = None

            teacher_results = lm_eval.simple_evaluate(
                model=teacher_lm,
                tasks=args.tasks,
                batch_size=args.batch_size,
                limit=limit,
                confirm_run_unsafe_code=True,
            )


    print("\n== student ==")
    print_results(student_results)
    if teacher_results is not None:
        print("\n== teacher ==")
        print_results(teacher_results)

    def _json_default(obj):
        try:
            return obj.item()
        except AttributeError:
            return str(obj)

    def _summarize(results: Mapping[str, dict]) -> dict:
        return {
            "results": results.get("results", {}),
            "aggregate": results.get("aggregate"),
            "config": results.get("config", {}),
        }

    payload = {"student": _summarize(student_results)}
    if teacher_results is not None:
        payload["teacher"] = _summarize(teacher_results)
    print("\nSaving results to", args.output)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


if __name__ == "__main__":
    main()
