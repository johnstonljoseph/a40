"""Utility to inspect diag scaling payloads without printing the raw scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

SCRIPT_DIR = Path(__file__).resolve().parent


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print diag-scaling payloads stored under the values/ directory, "
        "excluding the raw 'scales' tensors."
    )
    parser.add_argument(
        "--values-dir",
        type=Path,
        default=SCRIPT_DIR / "values",
        help="Directory containing per-layer payloads (default: %(default)s).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for payload output (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Only print the top-K maxima across all linear maps (default: %(default)s).",
    )
    return parser.parse_args()


def tensor_stats(tensor: torch.Tensor, *, use_abs: bool = False) -> dict[str, float]:
    data = tensor.detach().to(torch.float32)
    if use_abs:
        data = data.abs()
    return {
        "numel": data.numel(),
        "min": float(data.min().item()),
        "max": float(data.max().item()),
        "mean": float(data.mean().item()),
    }


def summarize_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload).__name__}")

    summary: dict[str, Any] = {}
    act = payload.get("act_percentile")
    weight = payload.get("weight_percentile")
    scales = payload.get("scales")

    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            if key == "scales":
                summary["scales_abs_stats"] = tensor_stats(value, use_abs=True)
            elif key not in ("act_percentile", "weight_percentile"):
                summary[f"{key}_stats"] = tensor_stats(value, use_abs=False)
        else:
            summary[key] = _to_serializable(value)

    if isinstance(act, torch.Tensor):
        summary["act_percentile_max"] = float(act.max().item())
    if isinstance(weight, torch.Tensor):
        summary["weight_percentile_max"] = float(weight.max().item())
    if isinstance(act, torch.Tensor) and isinstance(scales, torch.Tensor):
        if act.shape != scales.shape:
            raise ValueError(f"Shape mismatch for act_percentile ({tuple(act.shape)}) vs scales ({tuple(scales.shape)})")
        eps = torch.finfo(scales.dtype).eps
        act_over_scale = act / scales.clamp_min(eps)
        summary["_act_over_scale_array"] = act_over_scale
        max_act_over_scale_val, max_act_over_scale_idx = act_over_scale.max(dim=0)
    else:
        max_act_over_scale_val = max_act_over_scale_idx = None

    if isinstance(weight, torch.Tensor) and isinstance(scales, torch.Tensor):
        if weight.shape != scales.shape:
            raise ValueError(
                f"Shape mismatch for weight_percentile ({tuple(weight.shape)}) vs scales ({tuple(scales.shape)})"
            )
        weight_scaled = weight * scales
        summary["_weight_scaled_array"] = weight_scaled
        max_weight_scaled_val, max_weight_scaled_idx = weight_scaled.max(dim=0)
    else:
        max_weight_scaled_val = max_weight_scaled_idx = None

    if max_act_over_scale_val is not None and max_weight_scaled_val is not None:
        partner_weight = float(weight_scaled[max_act_over_scale_idx].item())
        partner_act_over_scale = float(act_over_scale[max_weight_scaled_idx].item())
        summary["max_act_over_scale"] = {
            "value": float(max_act_over_scale_val.item()),
            "partner_weight_scaled": partner_weight,
        }
        summary["max_weight_scaled"] = {
            "value": float(max_weight_scaled_val.item()),
            "partner_act_over_scale": partner_act_over_scale,
        }
        # cleanup raw arrays
        summary.pop("_act_over_scale_array", None)
        summary.pop("_weight_scaled_array", None)
    elif max_act_over_scale_val is not None:
        summary["max_act_over_scale"] = float(max_act_over_scale_val.item())
    elif max_weight_scaled_val is not None:
        summary["max_weight_scaled"] = float(max_weight_scaled_val.item())
    return summary


def extract_max_pairs(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload).__name__}")

    act = payload.get("act_percentile")
    weight = payload.get("weight_percentile")
    scales = payload.get("scales")

    if not (isinstance(act, torch.Tensor) and isinstance(weight, torch.Tensor) and isinstance(scales, torch.Tensor)):
        return []
    if act.shape != scales.shape or weight.shape != scales.shape:
        raise ValueError(
            "Shape mismatch: "
            f"act_percentile={tuple(act.shape)}, weight_percentile={tuple(weight.shape)}, scales={tuple(scales.shape)}"
        )

    eps = torch.finfo(scales.dtype).eps
    act_over_scale = act / scales.clamp_min(eps)
    weight_scaled = weight * scales

    max_aos_val, max_aos_idx = act_over_scale.max(dim=0)
    max_ws_val, max_ws_idx = weight_scaled.max(dim=0)

    entries: list[dict[str, Any]] = []
    entries.append(
        {
            "kind": "max_act_over_scale",
            "value": float(max_aos_val.item()),
            "A": float(act[max_aos_idx].item()),
            "W": float(weight[max_aos_idx].item()),
            "s": float(scales[max_aos_idx].item()),
            "A_over_s": float(act_over_scale[max_aos_idx].item()),
            "W_times_s": float(weight_scaled[max_aos_idx].item()),
        }
    )
    entries.append(
        {
            "kind": "max_weight_scaled",
            "value": float(max_ws_val.item()),
            "A": float(act[max_ws_idx].item()),
            "W": float(weight[max_ws_idx].item()),
            "s": float(scales[max_ws_idx].item()),
            "A_over_s": float(act_over_scale[max_ws_idx].item()),
            "W_times_s": float(weight_scaled[max_ws_idx].item()),
        }
    )
    return entries


def main() -> None:
    args = parse_args()
    values_dir: Path = args.values_dir
    _indent = int(args.indent)
    top_k = int(args.top_k)

    if not values_dir.exists():
        raise FileNotFoundError(f"{values_dir} does not exist.")
    if not values_dir.is_dir():
        raise NotADirectoryError(f"{values_dir} is not a directory.")

    payload_paths = sorted(values_dir.rglob("*.pt"))
    if not payload_paths:
        print(f"No payload files found under {values_dir}")
        return

    all_entries: list[dict[str, Any]] = []
    for payload_path in payload_paths:
        rel_path = payload_path.relative_to(values_dir)
        try:
            for entry in extract_max_pairs(payload_path):
                entry["linear_map"] = str(rel_path)
                all_entries.append(entry)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERROR] Failed to load {payload_path}: {exc}")

    if not all_entries:
        print("No eligible payloads found (expected act_percentile, weight_percentile, scales).")
        return

    all_entries.sort(key=lambda e: float(e["value"]), reverse=True)
    for i, entry in enumerate(all_entries[:top_k]):
        print(
            f"{i + 1:02d} {entry['linear_map']} {entry['kind']} "
            f"A={entry['A']:.6g} W={entry['W']:.6g} s={entry['s']:.6g} "
            f"A/s={entry['A_over_s']:.6g} W*s={entry['W_times_s']:.6g}"
        )


if __name__ == "__main__":
    main()
