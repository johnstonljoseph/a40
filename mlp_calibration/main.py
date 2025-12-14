import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from tqdm.auto import tqdm

from a40.main_relu import Config, load_model, resolve_model_path
from a40.data import build_dataloader

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect MLP pre-activation inputs per layer and solve a per-layer MSE calibration problem."
    )
    parser.add_argument("--model-path", type=str, default=Config.base_path)
    parser.add_argument("--batch_count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seq-len", type=int, default=Config.seq_len)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--dtype", type=str, default=Config.dtype)
    parser.add_argument(
        "--output",
        type=str,
        default=str(SCRIPT_DIR / "values.txt"),
        help="Output path for per-layer scalars.",
    )

    return parser.parse_args()


def register_mlp_act_input_hooks(
    model: torch.nn.Module,
) -> tuple[Dict[int, list[torch.Tensor]], list[torch.utils.hooks.RemovableHandle]]:
    activations: Dict[int, list[torch.Tensor]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_index, layer in enumerate(model.model.layers):
        activations[layer_index] = []
        act_fn = layer.mlp.act_fn

        def hook(_module, inputs, _output, *, _layer_index: int = layer_index):
            x = inputs[0].detach().float().reshape(-1)
            activations[_layer_index].append(x)

        handles.append(act_fn.register_forward_hook(hook))

    return activations, handles


def silu_and_grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.sigmoid(x)
    y = x * s
    dy = s + x * s * (1 - s)
    return y, dy


def piecewise_soft_and_grad(
    x: torch.Tensor,
    a_p: torch.Tensor,
    a_m: torch.Tensor,
    x0: torch.Tensor,
    k: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = torch.sigmoid(k * (x - x0))
    y = m * (a_p * (x - x0)) + (1.0 - m) * (a_m * (x - x0))
    dy = m * a_p + (1.0 - m) * a_m
    return y, dy


def solve_activation_fit(
    z: torch.Tensor,
    lam: float = 20.0,
    k: float = 20.0,
    max_iter: int = 1000,
) -> dict:
    z = z.detach().float()

    yT, dT = silu_and_grad(z)

    a_p = torch.nn.Parameter(z.new_tensor(1.0))
    a_m = torch.nn.Parameter(z.new_tensor(-0.1))
    x0 = torch.nn.Parameter(z.new_tensor(-0.2))

    opt = torch.optim.LBFGS(
        [a_p, a_m, x0],
        lr=1.0,
        max_iter=max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    w = torch.exp(-0.5 * z**2)

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        yS, dS = piecewise_soft_and_grad(z, a_p, a_m, x0, k=k)
        loss_d = (w * (dS - dT) ** 2).mean()
        loss_y = ((yS - yT) ** 2).mean()
        reg = 1e-3 * ((a_p - 1.0) ** 2 + (a_m + 0.2) ** 2 + x0**2)
        loss = loss_d + lam * loss_y + reg
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        yS, dS = piecewise_soft_and_grad(z, a_p, a_m, x0, k=k)
        final_loss = (dS - dT).pow(2).mean() + lam * (yS - yT).pow(2).mean()

    return {
        "a_p": float(a_p.item()),
        "a_m": float(a_m.item()),
        "x0": float(x0.item()),
        "final_loss": float(final_loss.item()),
    }


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    model_path = resolve_model_path(args.model_path)
    print(
        f"[mlp_calib] loading model from {model_path} "
        f"(device={device}, dtype={dtype})",
        flush=True,
    )
    model = load_model(model_path, device, dtype)

    activations, handles = register_mlp_act_input_hooks(model)

    cfg = Config(batch_size=args.batch_size, seq_len=args.seq_len, shuffle_buffer_size=0)
    dataloader = build_dataloader(
        cfg,
        model_path,
        world_size=1,
        rank=0,
    )
    batch_iter = iter(dataloader)

    print(
        f"[mlp_calib] streaming {args.batch_count} batches",
        flush=True,
    )

    with torch.no_grad():
        for _ in tqdm(range(args.batch_count), desc="mlp-calib", disable=False):
            batch = next(batch_iter)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

    for handle in handles:
        handle.remove()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for layer_index in sorted(activations.keys()):
            z = torch.cat(activations[layer_index], dim=0)
            result = solve_activation_fit(z)
            f.write(
                f"{layer_index}\t{result['a_p']:.10f}\t{result['a_m']:.10f}\t{result['x0']:.10f}\t{result['final_loss']:.10f}\n"
            )

    print(f"[mlp_calib] wrote {out_path} ({len(activations)} layers)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
