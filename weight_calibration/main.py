import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from a40.main import (
    Config,
    iter_layer_linears,
    load_model,
    resolve_model_path,
)
from .solve import solve

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute per-layer weight scales for QuantLinear.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=Config.base_path,
        help="Path (or HF snapshot dir) to the model to calibrate.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        help="Comma-separated decoder layer indices to process (e.g. 0,1,2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=Config.device,
        help="Device to load the model on (e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Torch dtype to load the model with (e.g., float32, bfloat16).",
    )
    args = parser.parse_args()
    raw_layers = args.layers.strip()
    args.layer_ids = tuple(int(tok) for tok in raw_layers.split(",") if tok.strip())
    return args


@torch.no_grad()
def solve_scales(weight: torch.Tensor, qmax: int) -> torch.Tensor:
    """Match QuantLinear.initialize(): per-row max divided by (qmax - 0.5)."""

    progress = tqdm(
        total=weight.shape[0],
        leave=False,
    )

    b = float(qmax) - 0.5
    scales = []
    for row in weight:
        scales.append(solve(row.abs(), b))
        progress.update(1)
    progress.close()

    return torch.stack(scales)


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    model_path = resolve_model_path(args.model_path)
    print(f"[calib] loading model from {model_path} (device={device}, dtype={dtype})", flush=True)
    model = load_model(model_path, device, dtype)

    bits = 8
    qmax = 1 << (bits - 1)

    total_written = 0
    for layer_index, _parent, name, child in iter_layer_linears(model.model.layers):
        if layer_index not in args.layer_ids:
            continue
        print(f"[calib] solving {name} (layer {layer_index})...", flush=True)
        scales = solve_scales(child.weight.detach(), qmax).cpu()
        layer_dir = SCRIPT_DIR / "values" / str(layer_index)
        layer_dir.mkdir(exist_ok=True)
        payload = {
            "bits": bits,
            "scales": scales,
        }
        path = layer_dir / f"{name}.pt"
        torch.save(payload, path)
        total_written += 1
        print(f"[calib] wrote {path} ({scales.numel()} values)")


if __name__ == "__main__":
    main()
