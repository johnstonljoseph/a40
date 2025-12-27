#!/usr/bin/env python3
"""
Dump per-layer activation parameters (a_p, a_m, x0, y0) from a saved checkpoint.

Example:
    python dump_activation_params.py --ckpt /workspace/a40/checkpoints/student_final/r5-250
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from safetensors.torch import safe_open


PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.act_fn\._(a_p|a_m|x0|y0)$")


def load_activation_params(ckpt_dir: Path) -> dict[int, dict[str, float]]:
    """Read activation parameters directly from safetensors shards without loading the full model."""
    index_path = ckpt_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")

    with index_path.open("r") as f:
        index_data = json.load(f)

    weight_map: dict[str, str] = index_data.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"No weight_map found in {index_path}")

    # Collect keys we care about and group by shard file.
    shard_to_keys: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for key, shard in weight_map.items():
        match = PATTERN.match(key)
        if not match:
            continue
        layer_idx = int(match.group(1))
        name = match.group(2)
        shard_to_keys[shard].append((key, layer_idx, name))

    if not shard_to_keys:
        raise ValueError("No activation parameters found in checkpoint.")

    per_layer: dict[int, dict[str, float]] = defaultdict(dict)
    for shard, keys in shard_to_keys.items():
        shard_path = ckpt_dir / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard referenced in index is missing: {shard_path}")

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key, layer_idx, name in keys:
                tensor = f.get_tensor(key)
                per_layer[layer_idx][name] = float(tensor.item())

    return per_layer


def main():
    parser = argparse.ArgumentParser(description="Dump activation parameters per layer.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to checkpoint directory (contains model.safetensors.index.json).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of a human-readable table.",
    )
    args = parser.parse_args()

    params = load_activation_params(args.ckpt)

    if args.json:
        # Ensure deterministic ordering
        ordered = {layer: params[layer] for layer in sorted(params.keys())}
        print(json.dumps(ordered, indent=2, sort_keys=True))
        return

    print(f"Found activation parameters for {len(params)} layers in {args.ckpt}")
    for layer_idx in sorted(params.keys()):
        layer_params = params[layer_idx]
        missing = {p for p in ("a_p", "a_m", "x0", "y0") if p not in layer_params}
        if missing:
            print(f"Layer {layer_idx:02d}: MISSING {', '.join(sorted(missing))}")
            continue
        print(
            f"Layer {layer_idx:02d}: "
            f"a_p={layer_params['a_p']:.6f}, "
            f"a_m={layer_params['a_m']:.6f}, "
            f"x0={layer_params['x0']:.6f}, "
            f"y0={layer_params['y0']:.6f}"
        )


if __name__ == "__main__":
    main()
