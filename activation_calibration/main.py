import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
from tqdm.auto import tqdm

import datasketches

from a40.main import Config, load_model, resolve_model_path
from a40.data import build_dataloader

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline activation calibration for QuantLinear.")
    parser.add_argument("--model-path", type=str, default=Config.base_path)
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated decoder layer indices or 'all' (default).",
    )
    parser.add_argument("--batch_count", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seq-len", type=int, default=Config.seq_len)
    parser.add_argument("--quantile", type=float, default=0.9999)
    parser.add_argument("--sketch-k", type=int, default=(1 << 16) - 1)
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


class ActivationCollector:
    def __init__(self, batch_count: int, quantile: float, sketch_k: int):
        self.quantile = quantile
        self.sketch = datasketches.kll_floats_sketch(sketch_k)
        self.max_abs: float = 0.0

    def update(self, tensor: torch.Tensor) -> None:
        flat = tensor.detach().abs()
        current_max = float(flat.max().item())
        if current_max > self.max_abs:
            self.max_abs = current_max
        cpu_flat = flat.flatten().to("cpu", dtype=torch.float32)
        self.sketch.update(cpu_flat.numpy())

    def get_clip_value(self) -> float:
        return self.sketch.get_quantile(self.quantile)


def register_collectors(model: torch.nn.Module, target_layers: Tuple[int, ...], args) -> Tuple[
    Dict[Tuple[int, str], ActivationCollector], list[torch.utils.hooks.RemovableHandle]
]:
    collectors = {}
    handles = []

    def make_hook(key: Tuple[int, str]):
        def hook(_module, inputs, _output):
            activations = inputs[0]
            collectors[key].update(activations)
        return hook

    for layer_index, layer in enumerate(model.model.layers):
        if layer_index not in target_layers:
            continue
        groups = [
            (layer.self_attn, "q_k_v"),
            (layer.self_attn, "o"),
            (layer.mlp, "gate_up"),
            (layer.mlp, "down"),
        ]
        for module, group_name in groups:
            key = (layer_index, group_name)
            collectors[key] = ActivationCollector(args.batch_count, args.quantile, args.sketch_k)

            first_suffix = group_name.split("_")[0]
            linear = getattr(module, f"{first_suffix}_proj")
            handles.append(linear.register_forward_hook(make_hook(key)))

    return collectors, handles


def main():
    args = parse_args()
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
            f"[calib] loading model from {model_path} "
            f"(device={device}, dtype={dtype}, world_size={world_size})",
            flush=True,
        )
    model = load_model(model_path, device, dtype)

    if args.layer_ids is None:
        args.layer_ids = tuple(range(len(model.model.layers)))

    assigned_layers = args.layer_ids[rank::world_size]
    if not assigned_layers:
        print(f"[calib][rank {rank}] no target layers assigned; exiting early.", flush=True)
        return
    print(f"[calib][rank {rank}] handling layers {assigned_layers}", flush=True)

    collectors, handles = register_collectors(model, assigned_layers, args)

    cfg = Config(batch_size=args.batch_size, seq_len=args.seq_len)
    dataloader = build_dataloader(
        cfg,
        model_path,
        world_size=world_size,
        rank=rank,
    )
    batch_iter = iter(dataloader)

    if rank == 0:
        print(f"[calib] streaming {args.batch_count} batches per rank...", flush=True)
    with torch.no_grad():
        for _ in tqdm(
            range(args.batch_count),
            desc=f"activation-calib[r{rank}]",
            disable=(rank != 0),
        ):
            batch = next(batch_iter)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

    for handle in handles:
        handle.remove()

    for (layer_index, group_name), collector in collectors.items():
        clip_value = collector.get_clip_value()
        layer_dir = SCRIPT_DIR / "values" / str(layer_index)
        layer_dir.mkdir(exist_ok=True)
        payload = {
            "quantile": args.quantile,
            "clip_value": clip_value,
            "max_abs": collector.max_abs,
        }
        path = layer_dir / f"{group_name}.pt"
        torch.save(payload, path)
        print(
            f"[calib][rank {rank}] wrote {layer_index}.{group_name} "
            f"(clip_value={clip_value:.6f}, max_abs={collector.max_abs:.6f})"
        )


if __name__ == "__main__":
    main()
