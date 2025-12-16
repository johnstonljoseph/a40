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
        "--layer-index",
        type=int,
        default=None,
        help="If set, only collect activations for this layer and write a histogram PNG instead of fitting all layers.",
    )
    parser.add_argument(
        "--hist-all",
        action="store_true",
        help="If set, write one annotated histogram PNG per layer and exit.",
    )
    parser.add_argument(
        "--hist-dir",
        type=str,
        default=str(SCRIPT_DIR / "hists"),
        help="Output directory for per-layer histogram PNGs (only used with --hist-all).",
    )
    parser.add_argument(
        "--hist-output",
        type=str,
        default=str(SCRIPT_DIR / "hist_layer.png"),
        help="Histogram PNG output path (only used when --layer-index is set).",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=400,
        help="Number of bins for histogram (only used when --layer-index is set).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2_000_000,
        help="Max number of activation samples to keep for inspection (only used when --layer-index is set).",
    )
    parser.add_argument(
        "--sample-per-call",
        type=int,
        default=100_000,
        help="Max samples to keep from a single hook call when collecting histograms.",
    )
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


def register_mlp_act_input_hooks_sampled(
    model: torch.nn.Module,
    max_samples: int,
    sample_per_call: int,
) -> tuple[Dict[int, list[torch.Tensor]], Dict[int, int], list[torch.utils.hooks.RemovableHandle]]:
    activations: Dict[int, list[torch.Tensor]] = {}
    counts: Dict[int, int] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_index, layer in enumerate(model.model.layers):
        activations[layer_index] = []
        counts[layer_index] = 0
        act_fn = layer.mlp.act_fn

        def hook(_module, inputs, _output, *, _layer_index: int = layer_index):
            remaining = int(max_samples) - int(counts[_layer_index])
            if remaining <= 0:
                return
            x = inputs[0].detach().float().reshape(-1)
            take = min(int(sample_per_call), remaining, int(x.numel()))
            if take <= 0:
                return
            if take < int(x.numel()):
                idx = torch.randint(0, int(x.numel()), (take,), device=x.device)
                x = x.index_select(0, idx)
            activations[_layer_index].append(x.cpu())
            counts[_layer_index] += int(take)

        handles.append(act_fn.register_forward_hook(hook))

    return activations, counts, handles


def register_single_mlp_act_input_hook(
    model: torch.nn.Module,
    layer_index: int,
) -> tuple[list[torch.Tensor], torch.utils.hooks.RemovableHandle]:
    layers = getattr(model, "model").layers
    if not (0 <= layer_index < len(layers)):
        raise ValueError(f"Invalid layer index {layer_index}; model has {len(layers)} layers")
    act_fn = layers[layer_index].mlp.act_fn
    activations: list[torch.Tensor] = []

    def hook(_module, inputs, _output):
        x = inputs[0].detach().float().reshape(-1)
        activations.append(x)

    handle = act_fn.register_forward_hook(hook)
    return activations, handle


def print_activation_stats(z: torch.Tensor) -> None:
    z = z.detach().float().reshape(-1)
    z_min = float(z.min().item())
    z_max = float(z.max().item())
    qs = torch.tensor([0.9, 0.99, 0.999], dtype=torch.float32, device=z.device)
    qv = torch.quantile(z, qs).cpu().tolist()
    print(f"[inspect] min={z_min:.6f} max={z_max:.6f}", flush=True)
    print(
        f"[inspect] p90={qv[0]:.6f} p99={qv[1]:.6f} p999={qv[2]:.6f}",
        flush=True,
    )


def format_activation_stats(z: torch.Tensor) -> str:
    z = z.detach().float().reshape(-1)
    z_min = float(z.min().item())
    z_max = float(z.max().item())
    qs = torch.tensor([0.9, 0.99, 0.999], dtype=torch.float32, device=z.device)
    qv = torch.quantile(z, qs).cpu().tolist()
    return (
        f"min {z_min:.4f}\n"
        f"max {z_max:.4f}\n"
        f"p90 {qv[0]:.4f}\n"
        f"p99 {qv[1]:.4f}\n"
        f"p999 {qv[2]:.4f}"
    )


def save_histogram_png(
    z: torch.Tensor,
    out_path: Path,
    bins: int,
    title: str,
    stats_text: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(
            f"[inspect] skipping histogram PNG (matplotlib unavailable): {e}",
            flush=True,
        )
        return

    z = z.detach().float().reshape(-1)
    qs = torch.tensor([0.001, 0.999], dtype=torch.float32, device=z.device)
    q_lo, q_hi = torch.quantile(z, qs).cpu().tolist()
    z_np = z.cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(z_np, bins=bins, range=(q_lo, q_hi), density=False)
    ax.set_title(title)
    ax.set_xlabel("activation value")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.2)
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


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

    if args.layer_index is not None and args.hist_all:
        raise ValueError("Use either --layer-index or --hist-all, not both")

    if args.layer_index is not None:
        layer_index = int(args.layer_index)
        activations_single, handle = register_single_mlp_act_input_hook(model, layer_index)
    elif args.hist_all:
        activations, counts, handles = register_mlp_act_input_hooks_sampled(
            model,
            max_samples=int(args.max_samples),
            sample_per_call=int(args.sample_per_call),
        )
    else:
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

            if args.layer_index is not None and len(activations_single) > 0:
                total = sum(int(x.numel()) for x in activations_single)
                if total >= int(args.max_samples):
                    break

    if args.layer_index is not None:
        handle.remove()
        if len(activations_single) == 0:
            raise RuntimeError("No activations collected (did the model run any forward passes?)")
        z = torch.cat(activations_single, dim=0)
        if z.numel() > int(args.max_samples):
            z = z[: int(args.max_samples)]
        print(f"[inspect] layer={args.layer_index} samples={int(z.numel())}", flush=True)
        print_activation_stats(z)
        save_histogram_png(
            z,
            Path(args.hist_output),
            bins=int(args.hist_bins),
            title=f"Layer {args.layer_index} act_fn input histogram (clipped to [p0.1%, p99.9%])",
            stats_text=format_activation_stats(z),
        )
        print(f"[inspect] wrote histogram: {args.hist_output}", flush=True)
        os._exit(0)
    elif args.hist_all:
        for handle in handles:
            handle.remove()
        out_dir = Path(args.hist_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        num_layers = len(activations)
        print(
            f"[inspect] writing histograms for {num_layers} layers to {out_dir} (max_samples={int(args.max_samples)})",
            flush=True,
        )
        for layer_index in sorted(activations.keys()):
            if len(activations[layer_index]) == 0:
                continue
            z = torch.cat(activations[layer_index], dim=0)
            if z.numel() > int(args.max_samples):
                z = z[: int(args.max_samples)]
            stats_text = format_activation_stats(z)
            out_path = out_dir / f"layer_{layer_index:02d}.png"
            save_histogram_png(
                z,
                out_path,
                bins=int(args.hist_bins),
                title=f"Layer {layer_index} act_fn input histogram (clipped to [p0.1%, p99.9%])",
                stats_text=stats_text,
            )
        print(f"[inspect] wrote histogram directory: {out_dir}", flush=True)
        os._exit(0)
    else:
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
