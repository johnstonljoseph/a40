#!/usr/bin/env python3
"""Plot max |activation| from a40 data stream for a given decoder layer.

- Uses the same tokenizer/chat rendering as a40.data.
- Loads the HF 1B Instruct snapshot (default) and runs a small number of batches.
- Captures the down_proj output and plots per-feature max |activation|.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import LlamaForCausalLM

from .data import build_dataloader
from .main import Config, resolve_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot down_proj max |activation| for a layer using a40 data stream.")
    parser.add_argument(
        "--base-path",
        type=str,
        default="/Users/joseph/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct",
        help="Path to HF model repo or snapshot (same as a40 Config.base_path).",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=1,
        help="Decoder layer index (0-based).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (default: cpu).",
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
    model = LlamaForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.eval().to(device)

    batch_iter = iter(build_dataloader(
        cfg,
        model_path,
        1,
        0,
    ))

    storage: List[torch.Tensor] = []

    def hook(_module, _inp, output: torch.Tensor) -> None:
        # capture output of gate_proj: shape [batch, seq, hidden]
        storage.append(output.detach().abs().amax(dim=(0, 1)).to("cpu", torch.float32))

    layer = model.model.layers[args.layer_index]
    handle = layer.mlp.up_proj.register_forward_hook(hook)

    try:
        for i in range(args.batches):
            print("next")
            batch = next(batch_iter)
            print("to device")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                print("begin")
                model(**batch, use_cache=False)
                print("end")
            print(f"ran batch {i+1}")
    finally:
        handle.remove()

    # Stack per-forward feature maxima: [num_forwards, hidden] -> max over forwards
    acts = torch.stack(storage, dim=0)
    feature_ids = np.arange(acts.shape[1])

    max_abs = acts.max(dim=0).values.numpy()

    top_idx = np.argpartition(-max_abs, min(10, len(max_abs) - 1))[:10]
    top_sorted = top_idx[np.argsort(-max_abs[top_idx])]
    print("Top features by max |activation|:")
    for rank, idx in enumerate(top_sorted, 1):
        print(f"{rank:2d}: feature {feature_ids[idx]} -> {max_abs[idx]:.6f}")

    plt.figure(figsize=(12, 5))
    plt.bar(feature_ids, max_abs, color="darkcyan")
    plt.xlabel("feature index")
    plt.ylabel("max |activation| across tokens")
    plt.title(f"down_proj max |activation| (layer {args.layer_index})")
    plt.tight_layout()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
