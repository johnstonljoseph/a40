import argparse
import importlib
import sys
from pathlib import Path
from tqdm.auto import tqdm
from .solve import solve

import torch

# Script is run via `python -m weight_calibration.main` from the a40 directory.
# Make sure the parent directory (which contains the `a40` namespace) is on sys.path
# so we can import `a40.main` and reuse its utilities.
repo_root = Path(__file__).resolve().parents[1]
project_parent = repo_root.parent
if str(project_parent) not in sys.path:
    sys.path.insert(0, str(project_parent))
main_module = importlib.import_module(f"{repo_root.name}.main")
Config = main_module.Config  # type: ignore[attr-defined]
_iter_layer_linears = main_module._iter_layer_linears  # type: ignore[attr-defined]
load_model = main_module.load_model  # type: ignore[attr-defined]
resolve_model_path = main_module.resolve_model_path  # type: ignore[attr-defined]


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
        "--output-dir",
        type=str,
        default="/workspace/src/a40/weight_scales",
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
    model.eval()

    qmax = 1 << (args.bits - 1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for layer_index, _parent, name, child in _iter_layer_linears(model.model.layers):
        if layer_index not in args.layer_ids:
            continue
        print(f"[calib] solving {name} (layer {layer_index})...", flush=True)
        scales = solve_scales(child.weight.detach(), qmax).cpu()
        layer_dir = output_dir / str(layer_index)
        layer_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "bits": args.bits,
            "scales": scales,
        }
        path = layer_dir / f"{name}.pt"
        torch.save(payload, path)
        total_written += 1
        print(f"[calib] wrote {path} ({scales.numel()} values)")

    print(f"[calib] completed: {total_written} files in {output_dir}")


if __name__ == "__main__":
    main()
