import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

import datasketches

from a40.main import Config, iter_layer_linears, load_model, resolve_model_path
from a40.data import build_dataloader

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect SmoothQuant diagonal scaling factors per linear layer."
    )
    parser.add_argument("--model-path", type=str, default=Config.base_path)
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated decoder layer indices or 'all'.",
    )
    parser.add_argument("--batch-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seq-len", type=int, default=Config.seq_len)
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.9999,
        help="Percentile (0-1] for |activation| and |weight| statistics.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="SmoothQuant alpha (0-1).",
    )
    parser.add_argument(
        "--sketch-k",
        type=int,
        default=(1 << 16) - 1,
        help="KLL sketch size for streaming activation percentile estimation.",
    )
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--dtype", type=str, default=Config.dtype)
    args = parser.parse_args()
    raw_layers = args.layers.strip()
    args.layer_ids: Optional[Tuple[int, ...]] = (
        None
        if raw_layers.lower() == "all"
        else tuple(int(tok) for tok in raw_layers.split(",") if tok.strip())
    )
    return args


class ChannelPercentileCollector:
    def __init__(self, sketch_k: int) -> None:
        self._sketch_k = int(sketch_k)
        self._sketches: Optional[List[datasketches.kll_floats_sketch]] = None

    def update(self, tensor: torch.Tensor) -> None:
        flat = tensor.detach().abs().to(torch.float32).reshape(-1, tensor.shape[-1]).to("cpu")
        if self._sketches is None:
            self._sketches = [datasketches.kll_floats_sketch(self._sketch_k) for _ in range(flat.shape[-1])]

        np_flat = flat.numpy()
        for j, sketch in enumerate(self._sketches):
            sketch.update(np_flat[:, j])

    def percentile(self, q: float) -> torch.Tensor:
        values = [sketch.get_quantile(q) for sketch in self._sketches]
        return torch.tensor(values, dtype=torch.float32)


def register_collectors(
    model: torch.nn.Module,
    target_layers: Tuple[int, ...],
    sketch_k: int,
) -> Tuple[
    Dict[Tuple[int, str], ChannelPercentileCollector],
    Dict[Tuple[int, str], torch.nn.Linear],
    List[torch.utils.hooks.RemovableHandle],
]:
    collectors: Dict[Tuple[int, str], ChannelPercentileCollector] = {}
    modules: Dict[Tuple[int, str], torch.nn.Linear] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for layer_index, _parent, name, child in iter_layer_linears(model.model.layers):
        if name != "down_proj":
            # Only collect stats for down projections for now.
            continue
        if layer_index not in target_layers:
            continue
        key = (layer_index, name)
        collectors[key] = ChannelPercentileCollector(sketch_k)
        modules[key] = child

        def hook(_module, inputs, _output, *, _key=key):
            collectors[_key].update(inputs[0])

        handles.append(child.register_forward_hook(hook))

    return collectors, modules, handles


def smoothed_scale(
    act_percentile: torch.Tensor,
    weight_percentile: torch.Tensor,
    alpha: float,
    eps: float,
) -> torch.Tensor:
    act_term = (
        torch.pow(act_percentile + eps, alpha)
        if alpha != 0.0
        else torch.ones_like(act_percentile)
    )
    beta = 1.0 - alpha
    weight_term = (
        torch.pow(weight_percentile + eps, beta)
        if beta != 0.0
        else torch.ones_like(weight_percentile)
    )
    return act_term / weight_term


def main() -> None:
    args = parse_args()

    percentile = float(args.percentile)
    if not (0.0 < percentile <= 1.0):
        raise ValueError(f"percentile must be in (0, 1], got {percentile}")
    alpha = float(args.alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    device_str = args.device
    if device_str.startswith("cuda"):
        device_str = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    device = torch.device(device_str)
    dtype = getattr(torch, args.dtype)

    model_path = resolve_model_path(args.model_path)
    if rank == 0:
        print(
            f"[diag-scale] loading model from {model_path} "
            f"(device={device}, dtype={dtype}, world_size={world_size})",
            flush=True,
        )
    model = load_model(model_path, device, dtype)

    if args.layer_ids is None:
        args.layer_ids = tuple(range(len(model.model.layers)))

    assigned_layers = args.layer_ids[rank::world_size]
    if not assigned_layers:
        print(f"[diag-scale][rank {rank}] no target layers assigned.", flush=True)
        return
    print(f"[diag-scale][rank {rank}] handling layers {assigned_layers}", flush=True)

    collectors, modules, handles = register_collectors(model, tuple(assigned_layers), args.sketch_k)

    cfg = Config(batch_size=args.batch_size, seq_len=args.seq_len)
    dataloader = build_dataloader(
        cfg,
        model_path,
        world_size=world_size,
        rank=rank,
    )
    batch_iter = iter(dataloader)

    with torch.no_grad():
        for _ in tqdm(
            range(args.batch_count),
            desc=f"diag-scale[r{rank}]",
            disable=(rank != 0),
        ):
            batch = next(batch_iter)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

    for handle in handles:
        handle.remove()

    eps = torch.finfo(torch.float32).eps
    total_written = 0
    for key in sorted(collectors.keys()):
        layer_index, name = key
        act_percentiles = collectors[key].percentile(percentile)
        weight_abs = modules[key].weight.detach().to(torch.float32).abs()
        weight_percentiles = torch.quantile(weight_abs, percentile, dim=0)
        act_percentiles = act_percentiles.to(weight_percentiles.device)
        scales = smoothed_scale(act_percentiles, weight_percentiles, alpha, eps).cpu()

        layer_dir = SCRIPT_DIR / "values" / str(layer_index)
        layer_dir.mkdir(parents=True, exist_ok=True)
        path = layer_dir / f"{name}.pt"
        payload = {
            "alpha": alpha,
            "percentile": percentile,
            "act_percentile": act_percentiles.cpu(),
            "weight_percentile": weight_percentiles.cpu(),
            "scales": scales,
        }
        torch.save(payload, path)
        total_written += 1
        print(
            f"[diag-scale][rank {rank}] wrote {path} (n_channels={scales.numel()})",
            flush=True,
        )

    if rank == 0:
        print(f"[diag-scale] finished ({total_written} tensors)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
