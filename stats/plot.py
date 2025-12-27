#!/usr/bin/env python3
"""Capture max |activation| outputs for every Linear map across all decoder layers."""

from __future__ import annotations

import argparse
import os
import sys
import signal
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.models.olmo3 import Olmo3ForCausalLM

from .data import build_dataloader
from .main import Config, resolve_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture max |activation| outputs for every Linear map across all decoder layers.")
    parser.add_argument(
        "--base-path",
        type=str,
        default="/workspace/.hf_home/hub/models--allenai--Olmo-3-7B-Think",
        help="Path to HF model repo or snapshot (same as a40 Config.base_path).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Torch dtype name (default: float32).",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=12,
        help="Number of batches to accumulate max over (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the plot (otherwise shows).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    cfg = Config(
        base_path=args.base_path,
        device=args.device,
        dtype=args.dtype,
        train_layers=(),
        num_workers=0,
    )

    model_path = resolve_model_path(cfg.base_path)
    print(f"Loading model from {model_path} on {device}...")
    prev_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        model = Olmo3ForCausalLM.from_pretrained(model_path, dtype=dtype)
    except KeyboardInterrupt:
        print("Interrupted while loading model.", flush=True)
        return
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
    model.eval().to(device)

    batch_iter = iter(build_dataloader(
        cfg,
        model_path,
        1,
        0,
    ))

    # Dictionaries to store max values across all batches
    module_out_data: dict[str, float] = {}  # Linear outputs
    module_in_data: dict[str, float] = {}   # Linear inputs
    rmsnorm_data: dict[str, float] = {}     # RMSNorm inputs (pre-hook)
    residual_data: dict[str, float] = {}    # Residual stream entering each layer
    handles = []

    def linear_pre_hook_factory(module_name: str) -> callable:
        def pre_hook(_module, inputs) -> None:
            if not inputs:
                return
            tensor = inputs[0]
            if tensor is None:
                return
            max_val = tensor.detach().abs().amax().item()
            if module_name in module_in_data:
                module_in_data[module_name] = max(module_in_data[module_name], max_val)
            else:
                module_in_data[module_name] = max_val
        return pre_hook

    def linear_hook_factory(module_name: str) -> callable:
        def hook(_module, _inp, output: torch.Tensor) -> None:
            # capture max abs activation for this module across batch and sequence
            max_val = output.detach().abs().amax().item()
            # Update max for this module across batches
            if module_name in module_out_data:
                module_out_data[module_name] = max(module_out_data[module_name], max_val)
            else:
                module_out_data[module_name] = max_val
        return hook

    def rms_pre_hook_factory(module_name: str) -> callable:
        def pre_hook(_module, inputs) -> None:
            # inputs is a tuple; take first tensor
            if not inputs:
                return
            tensor = inputs[0]
            if tensor is None:
                return
            max_val = tensor.detach().abs().amax().item()
            if module_name in rmsnorm_data:
                rmsnorm_data[module_name] = max(rmsnorm_data[module_name], max_val)
            else:
                rmsnorm_data[module_name] = max_val
        return pre_hook

    # Register hooks for every Linear-like module inside each decoder layer
    for layer_idx, layer in enumerate(model.model.layers):
        # Residual entering the layer (before any submodules)
        handles.append(layer.register_forward_pre_hook(
            lambda _m, inputs, idx=layer_idx: residual_data.__setitem__(
                f"layer{idx}.residual_in",
                max(residual_data.get(f"layer{idx}.residual_in", 0.0),
                    inputs[0].detach().abs().amax().item() if inputs and inputs[0] is not None else 0.0)
            )
        ))

        for subname, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                name = f"layer{layer_idx}.{subname or 'linear'}"
                handles.append(module.register_forward_pre_hook(linear_pre_hook_factory(name)))
                handles.append(module.register_forward_hook(linear_hook_factory(name)))

        # RMSNorm pre-hooks (inputs to norms)
        for norm_name in ["input_layernorm", "post_attention_layernorm", "post_feedforward_layernorm"]:
            norm_module = getattr(layer, norm_name, None)
            if norm_module is not None and isinstance(norm_module, nn.Module):
                name = f"layer{layer_idx}.{norm_name}"
                handles.append(norm_module.register_forward_pre_hook(rms_pre_hook_factory(name)))

    # Optional: include lm_head if present
    if isinstance(getattr(model, "lm_head", None), nn.Linear):
        handles.append(model.lm_head.register_forward_pre_hook(linear_pre_hook_factory("lm_head")))
        handles.append(model.lm_head.register_forward_hook(linear_hook_factory("lm_head")))

    try:
        for _ in tqdm(range(args.batches), desc="Processing batches"):
            try:
                batch = next(batch_iter)
            except StopIteration:
                print("[data] stream exhausted before requested batches; stopping early.")
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    # Get top modules by max activation (inputs and outputs)
    entries: list[tuple[str, str, float]] = []
    for name, val in module_in_data.items():
        entries.append((name, "input", val))
    for name, val in module_out_data.items():
        entries.append((name, "output", val))

    if entries:
        sorted_entries = sorted(entries, key=lambda x: x[2], reverse=True)
        limit = min(20, len(sorted_entries))
        print(f"Captured {len(module_out_data)} Linear-like modules; showing top {limit} (input/output):", flush=True)
        for rank, (module_name, kind, max_val) in enumerate(sorted_entries[:limit], 1):
            print(f"{rank:2d}: {module_name} [{kind}] -> {max_val:.6f}")
    else:
        print("No Linear activations captured (module_data empty). Check hook registration.", flush=True)

    # RMSNorm stats
    if rmsnorm_data:
        print("\nRMSNorm max |input| values (pre-hook), per layer:", flush=True)
        def layer_idx(name: str) -> int:
            try:
                return int(name.split(".")[0].replace("layer", ""))
            except Exception:
                return 10_000
        for name in sorted(rmsnorm_data.keys(), key=layer_idx):
            print(f"{name}: {rmsnorm_data[name]:.6f}")

    # Residual stream stats
    if residual_data:
        print("\nResidual stream max |value| entering each layer:", flush=True)
        for name in sorted(residual_data.keys(), key=lambda n: int(n.split('.')[0].replace('layer', ''))):
            print(f"{name}: {residual_data[name]:.6f}")

    # Optional plotting
    if args.output:
        module_names = list(module_data.keys())
        values = list(module_data.values())
        
        plt.figure(figsize=(15, 8))
        plt.barh(range(len(values)), values, color="darkorange")
        plt.yticks(range(len(values)), module_names, fontsize=8)
        plt.xlabel("max |activation|")
        plt.title("Max |activation| across all Linear modules")
        plt.gca().invert_yaxis()  # Highest values at top
        plt.tight_layout()
        
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")

    # Flush and hard-exit to avoid HF dataset background thread finalization crash
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
