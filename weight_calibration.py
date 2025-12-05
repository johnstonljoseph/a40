import argparse
from pathlib import Path

import torch

from .main import Config, load_model, resolve_model_path, _iter_layer_linears


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute per-layer weight scales for QuantLinear.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=Config.student_model_path,
        help="Path (or HF snapshot dir) to the model to calibrate.",
    )
    parser.add_argument(
        "--train-layers",
        type=str,
        required=True,
        help="Comma-separated decoder layer indices to process (e.g. 0,1,2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weight_scales",
        help="Directory where per-layer .pt files will be written.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        help="Quantization bit-width (must match QuantLinear configuration).",
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
        default=Config.dtype,
        help="Torch dtype to load the model with (e.g., float32, bfloat16).",
    )
    args = parser.parse_args()
    raw_layers = args.train_layers.strip()
    if not raw_layers:
        raise ValueError("--train-layers requires at least one integer index")
    try:
        layer_ids = tuple(int(tok) for tok in raw_layers.split(",") if tok.strip())
    except ValueError as exc:
        raise ValueError("--train-layers must be a comma-separated list of integers") from exc
    if not layer_ids:
        raise ValueError("--train-layers requires at least one integer index")
    args.layer_ids = layer_ids
    return args


@torch.no_grad()
def compute_scales(weight: torch.Tensor, qmax: int) -> torch.Tensor:
    """Match QuantLinear.initialize(): per-row max divided by (qmax - 0.5)."""
    b = float(qmax) - 0.5
    per_row_max = weight.abs().amax(dim=1).clamp_min(1e-12)
    return per_row_max / b


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    model_path = resolve_model_path(args.model_path)
    print(f"[calib] loading model from {model_path} (device={device}, dtype={dtype})", flush=True)
    model = load_model(model_path, device, dtype)
    model.eval()

    qmax = 1 << (args.bits - 1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for layer_index, _, name, child in _iter_layer_linears(model.model.layers):
        if layer_index not in args.layer_ids:
            continue
        scales = compute_scales(child.weight.detach(), qmax).cpu()
        payload = {
            "layer": layer_index,
            "name": name,
            "bits": args.bits,
            "scales": scales,
        }
        path = output_dir / f"layer{layer_index}_{name}.pt"
        torch.save(payload, path)
        total_written += 1
        print(f"[calib] wrote {path} ({scales.numel()} values)")

    if total_written == 0:
        raise ValueError("No layers matched --train-layers; nothing was written.")
    print(f"[calib] completed: {total_written} files in {output_dir}")


if __name__ == "__main__":
    main()
