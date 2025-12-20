#!/usr/bin/env python3
"""
Run a forward pass through a checkpoint and report clipping percent in QuantLinearWithWeights activations.
Clipping is measured on the pre-rounding values y = x / act_scale against [-qmax, qmax-1].
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import datasets
from transformers import AutoModelForCausalLM

# Ensure repo root is on sys.path for `a40` imports when run from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Registers custom classes (PiecewiseActivation, QuantLinearWithWeights) via side effect.
import a40.custom_model  # noqa: F401
from a40.quant import QuantLinearWithWeights
from a40.data import build_dataloader
from a40.utils import resolve_model_path


def register_clipping_hooks(model: torch.nn.Module) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"clipped": 0, "total": 0, "max_ratio": 0.0}
    )

    def make_hook(name: str, module: QuantLinearWithWeights):
        qmax = module.qmax
        limit = float(qmax - 1)

        def hook(mod, inputs, output):
            (x,) = inputs
            act_scale = mod.log_act_s.exp().to(dtype=x.dtype, device=x.device)
            y = x / act_scale
            clipped = ((y < -qmax) | (y > (qmax - 1))).sum().item()
            total = y.numel()
            max_abs = y.abs().max().item()
            ratio = max_abs / limit if limit > 0 else 0.0
            stats[name]["clipped"] += int(clipped)
            stats[name]["total"] += int(total)
            stats[name]["max_ratio"] = max(stats[name]["max_ratio"], ratio)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, QuantLinearWithWeights):
            module.register_forward_hook(make_hook(name, module))
    return stats


def format_stats(stats: Dict[str, Dict[str, int]]) -> str:
    rows = []
    for name, rec in stats.items():
        total = rec["total"]
        clipped = rec["clipped"]
        pct = 100.0 * clipped / total if total > 0 else 0.0
        rows.append((rec["max_ratio"], pct, name, clipped, total))
    rows.sort(reverse=True)
    lines = ["max_ratio | pct_clipped | clipped / total | module"]
    for ratio, pct, name, clipped, total in rows:
        lines.append(f"{ratio:8.4f} | {pct:8.4f}% | {clipped:7d} / {total:7d} | {name}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint directory.")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("/workspace/.hf_home/hub/models--allenai--Olmo-3-7B-Think"),
        help="Base model path (or snapshot) to load tokenizer from.",
    )
    parser.add_argument(
        "--dataset-sft",
        type=str,
        required=True,
        help="HF dataset name (streaming) with 'messages' field to draw prompts from (e.g., allenai/Dolci-Think-SFT-7B).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to draw when using --dataset-sft.",
    )
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for the prompt.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="torch dtype, e.g., bfloat16 or float16")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for prompts.")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)

    stats = register_clipping_hooks(model)

    class _Cfg:
        # minimal fields used in data.build_dataloader
        def __init__(self):
            self.seq_len = args.seq_len
            self.dataset_sft = args.dataset_sft
            self.dataset_dpo = None
            self.dataset_rl = None
            self.dataset_ratio_sft = 1.0
            self.dataset_ratio_dpo = 0.0
            self.dataset_ratio_rl = 0.0
            self.shuffle_buffer_size = 10000
            self.batch_size = args.batch_size
            self.num_workers = 0

    cfg = _Cfg()
    dataloader = build_dataloader(
        cfg,
        tokenizer_path=str(resolve_model_path(str(args.base_path))),
        seed=0,
        world_size=1,
        rank=0,
    )

    seen = 0
    for batch in dataloader:
        batch = {k: (v if torch.is_tensor(v) else torch.tensor(v)) for k, v in batch.items()}
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            model(**batch)
        seen += batch["input_ids"].shape[0]
        if seen >= args.num_samples:
            break

    print(format_stats(stats))


if __name__ == "__main__":
    main()
