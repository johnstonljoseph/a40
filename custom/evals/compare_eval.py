#!/usr/bin/env python3
import argparse
import json
from typing import Dict, Iterable, List, Optional, Tuple


# Preferred metric keys per task, followed by global fallbacks.
PREFERRED_KEYS: Dict[str, List[str]] = {
    "arc_challenge": ["acc_norm,none", "acc,none"],
    "hellaswag": ["acc_norm,none", "acc,none"],
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract"],
    "humaneval": ["pass@1,create_test"],
    "mbpp": ["pass_at_1,none"],
    "mmlu": ["acc,none"],
}
GLOBAL_PREF_ORDER: List[str] = [
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "pass@1,create_test",
    "pass_at_1,none",
]

# Core tasks we want to show explicitly.
KEY_TASKS: List[str] = [
    "arc_challenge",
    "gsm8k",
    "hellaswag",
    "humaneval",
    "mbpp",
    "mmlu",
    "mmlu_humanities",
    "mmlu_social_sciences",
    "mmlu_stem",
    "mmlu_other",
]

MMLU_GROUPS: List[str] = [
    "mmlu",
    "mmlu_humanities",
    "mmlu_social_sciences",
    "mmlu_stem",
    "mmlu_other",
]


def is_number(val) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def choose_metric(task: str, task_res: Dict) -> Optional[str]:
    for key in PREFERRED_KEYS.get(task, []) + GLOBAL_PREF_ORDER:
        if is_number(task_res.get(key)):
            return key
    for key, val in task_res.items():
        if is_number(val):
            return key
    return None


def choose_stderr_key(metric: str, task_res: Dict) -> Optional[str]:
    if metric is None:
        return None
    if "," in metric:
        name, rest = metric.split(",", 1)
        candidate = f"{name}_stderr,{rest}"
        if is_number(task_res.get(candidate)):
            return candidate
    candidate_simple = f"{metric}_stderr"
    if is_number(task_res.get(candidate_simple)):
        return candidate_simple
    # Fallback: first numeric stderr-like key.
    for key, val in task_res.items():
        if "stderr" in key and is_number(val):
            return key
    return None


def resolve_task_metric(
    task: str, student: Dict, teacher: Dict
) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
    metric = choose_metric(task, student) or choose_metric(task, teacher)
    if metric is None:
        return None, None, None, None, None

    s_val = float(student[metric]) if is_number(student.get(metric)) else None
    t_val = float(teacher[metric]) if is_number(teacher.get(metric)) else None

    s_stderr_key = choose_stderr_key(metric, student)
    t_stderr_key = choose_stderr_key(metric, teacher)
    s_stderr = float(student[s_stderr_key]) if s_stderr_key and is_number(student.get(s_stderr_key)) else None
    t_stderr = float(teacher[t_stderr_key]) if t_stderr_key and is_number(teacher.get(t_stderr_key)) else None

    return metric, s_val, t_val, s_stderr, t_stderr


def format_pct_err(val: Optional[float], stderr: Optional[float], mult: float) -> str:
    if val is None:
        return "-"
    main = 100 * val
    if stderr is None:
        return f"{main:6.2f}%"
    err = 100 * stderr * mult
    return f"{main:6.2f}% \u00b1{err:5.2f}%"


def render_table(headers: List[str], rows: List[List[str]], align: Optional[List[str]] = None) -> str:
    if not rows:
        return ""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    if align is None:
        align = ["left"] + ["right"] * (len(headers) - 1)

    def _fmt(cell: str, width: int, a: str) -> str:
        return cell.ljust(width) if a == "left" else cell.rjust(width)

    lines = []
    lines.append(" | ".join(_fmt(h, w, a) for h, w, a in zip(headers, widths, align)))
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(_fmt(c, w, a) for c, w, a in zip(row, widths, align)))
    return "\n".join(lines)


def make_section(
    title: str,
    tasks: Iterable[str],
    student_res: Dict,
    teacher_res: Dict,
    ci_mult: float,
) -> str:
    rows: List[List[str]] = []
    for task in tasks:
        s_task = student_res.get(task, {})
        t_task = teacher_res.get(task, {})
        metric, s_val, t_val, s_stderr, t_stderr = resolve_task_metric(task, s_task, t_task)
        if metric is None:
            continue
        delta = None
        if s_val is not None and t_val is not None:
            delta = s_val - t_val
        rows.append(
            [
                task,
                metric,
                format_pct_err(s_val, s_stderr, ci_mult),
                format_pct_err(t_val, t_stderr, ci_mult),
                f"{100 * delta:+6.2f}%" if delta is not None else "-",
            ]
        )
    if not rows:
        return f"{title}\n(no data)\n"
    table = render_table(
        ["Task", "Metric", f"Student (±{ci_mult}·stderr)", f"Teacher (±{ci_mult}·stderr)", "Δ (S-T)"],
        rows,
        align=["left", "left", "right", "right", "right"],
    )
    return f"{title}\n{table}\n"


def top_deltas(student_res: Dict, teacher_res: Dict, limit: int = 10, ci_mult: float = 1.0) -> str:
    rows = []
    all_tasks = set(student_res) | set(teacher_res)
    for task in all_tasks:
        if task == "aggregate":
            continue
        metric, s_val, t_val, s_stderr, t_stderr = resolve_task_metric(
            task, student_res.get(task, {}), teacher_res.get(task, {})
        )
        if metric is None or s_val is None or t_val is None:
            continue
        delta = s_val - t_val
        rows.append(
            (
                abs(delta),
                [
                    task,
                    metric,
                    format_pct_err(s_val, s_stderr, ci_mult),
                    format_pct_err(t_val, t_stderr, ci_mult),
                    f"{100 * delta:+6.2f}%",
                ],
            )
        )
    rows.sort(key=lambda x: x[0], reverse=True)
    trimmed = [r for _, r in rows[:limit]]
    if not trimmed:
        return ""
    table = render_table(
        ["Task", "Metric", f"Student (±{ci_mult}·stderr)", f"Teacher (±{ci_mult}·stderr)", "Δ (S-T)"],
        trimmed,
        align=["left", "left", "right", "right", "right"],
    )
    return f"Top deltas (abs) — first {limit}\n{table}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare student vs teacher eval results.")
    parser.add_argument(
        "results_path",
        nargs="?",
        default="eval_results.json",
        help="Path to eval_results.json produced by a40/eval.py",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many rows to show in the absolute-delta section.",
    )
    parser.add_argument(
        "--ci-mult",
        type=float,
        default=1.0,
        help="Multiplier for stderr to show alongside percentages (e.g., 1.96 for ~95%% CI).",
    )
    parser.add_argument(
        "--mmlu-only",
        action="store_true",
        help="If set, only show the MMLU aggregate buckets plus top deltas.",
    )
    args = parser.parse_args()

    with open(args.results_path) as f:
        payload = json.load(f)

    student_res = payload.get("student", {}).get("results", {})
    teacher_res = payload.get("teacher", {}).get("results", {})

    sections = []
    tasks_primary = MMLU_GROUPS if args.mmlu_only else KEY_TASKS
    sections.append(
        make_section(
            "Core tasks (stderr shown)",
            tasks_primary,
            student_res,
            teacher_res,
            ci_mult=args.ci_mult,
        )
    )
    delta_section = top_deltas(student_res, teacher_res, limit=args.top, ci_mult=args.ci_mult)
    if delta_section:
        sections.append(delta_section)

    print("\n".join(sections))


if __name__ == "__main__":
    main()
