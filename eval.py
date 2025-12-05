import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import lm_eval  # pip install lm-eval
from lm_eval.models.huggingface import HFLM

_lm_eval_root = Path(lm_eval.__file__).resolve().parents[1]
os.environ.setdefault("GIT_WORK_TREE", str(_lm_eval_root)); os.environ.setdefault("GIT_DIR", str(_lm_eval_root / ".git"))


def evaluate_model(
    model_or_name,
    *,
    tasks: Sequence[str],
    batch_size: int,
    num_fewshot: int = 0,
    limit: float | None = None,
    model_args: str | None = None,
):
    if not isinstance(model_or_name, str):
        model_or_name = HFLM(pretrained=model_or_name, batch_size=batch_size)
        model_args = None

    return lm_eval.simple_evaluate(
        model=model_or_name,
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="hf",
        help='lm-eval model type, e.g. "hf", "vllm", "openai"',
    )
    parser.add_argument(
        "--model_args",
        type=str,
        required=True,
        help=(
            'lm-eval model_args string, e.g. '
            '"pretrained=meta-llama/Meta-Llama-3-8B-Instruct,device=cuda:0,trust_remote_code=True"'
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="leaderboard_results.json",
        help="Where to save the aggregated results JSON",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional fraction of examples per task for quick debug, e.g. 0.01",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["gsm8k", "mmlu_pro", "gpqa_main"],
        help="Whitespace-separated lm-eval task names (e.g. gsm8k truthfulqa_mc1).",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=1,
        help="Per-device batch size passed through to lm-eval.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Few-shot examples to use (Open LLM LB typically uses 0 or small n).",
    )
    args = parser.parse_args()

    results = run_evaluation(
        args.model,
        model_args=args.model_args,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    print_results(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
