#!/usr/bin/env python3
"""
Dump QuantLinearWithWeights quantization params (log_act_s, log_weight_s) per module.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Registers custom classes
import a40.custom_model  # noqa: F401
from a40.quant import QuantLinearWithWeights


def dump_params(model) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantLinearWithWeights):
            out[name] = float(module.log_act_s.detach().cpu())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint directory.")
    parser.add_argument("--dtype", type=str, default="float32", help="torch dtype for load (e.g., float32)")
    parser.add_argument("--output", type=Path, help="Optional JSON output file.")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    )

    payload = dump_params(model)
    import json

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
