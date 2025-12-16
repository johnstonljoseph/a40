import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers.models.olmo3.modeling_olmo3 import ALL_ATTENTION_FUNCTIONS, repeat_kv

from a40.main_relu import Config, load_model, resolve_model_path
from a40.data import build_dataloader
from a40.mlp_calibration.main import (
    format_activation_stats,
    print_activation_stats,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def save_histogram_png_with_xlabel(
    z: torch.Tensor,
    out_path: Path,
    bins: int,
    title: str,
    stats_text: str,
    xlabel: str,
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
    ax.set_xlabel(xlabel)
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


def save_attention_token_full_vector_png(
    logits_row: torch.Tensor,
    out_path: Path,
    title: str,
    value: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(
            f"[inspect] skipping example full plot PNG (matplotlib unavailable): {e}",
            flush=True,
        )
        return

    logits_row = logits_row.detach().float().reshape(-1)
    logits_row = torch.nan_to_num(logits_row, neginf=-1e9, posinf=1e9)

    if value == "probs":
        x = logits_row - logits_row.max()
        y = torch.exp(x)
        y = y / y.sum()
        ylabel = "attention probability"
    elif value == "logits":
        y = logits_row
        ylabel = "attention logit (pre-softmax)"
    else:
        raise ValueError(value)

    y_np = y.cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    ax.bar(range(len(y_np)), y_np, width=1.0)
    ax.set_title(title)
    ax.set_xlabel("key position")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def save_attention_token_distribution_png(
    logits_row: torch.Tensor,
    out_path: Path,
    title: str,
    topn: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(
            f"[inspect] skipping example plot PNG (matplotlib unavailable): {e}",
            flush=True,
        )
        return

    logits_row = logits_row.detach().float().reshape(-1)
    logits_row = torch.nan_to_num(logits_row, neginf=-1e9, posinf=1e9)
    x = logits_row - logits_row.max()
    probs = torch.exp(x)
    probs = probs / probs.sum()

    n = int(min(int(topn), int(probs.numel())))
    vals, _idx = torch.topk(probs, k=n, dim=0)
    vals_np = vals.cpu().numpy()
    cum_np = vals.cumsum(dim=0).cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    ax[0].plot(vals_np, linewidth=1.2)
    ax[0].set_title(f"Top-{n} attention probabilities")
    ax[0].set_xlabel("rank")
    ax[0].set_ylabel("probability")
    ax[0].grid(True, alpha=0.2)

    ax[1].plot(cum_np, linewidth=1.2)
    ax[1].set_title(f"Cumulative mass of top-{n}")
    ax[1].set_xlabel("rank")
    ax[1].set_ylabel("cumulative probability")
    ax[1].set_ylim(0.0, 1.0)
    ax[1].grid(True, alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect attention pre-softmax logits for a single (layer, head) and save a histogram PNG."
    )
    parser.add_argument("--model-path", type=str, default=Config.base_path)
    parser.add_argument("--batch_count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seq-len", type=int, default=Config.seq_len)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--dtype", type=str, default=Config.dtype)

    parser.add_argument("--layer-index", type=int, required=True)
    parser.add_argument("--head-index", type=int, required=True)
    parser.add_argument(
        "--metric",
        type=str,
        default="avg_gap",
        choices=("avg_gap", "row_max", "topk_mass", "logits"),
        help=(
            "Value to collect: avg_gap=(top1-top2) per token (query position); "
            "row_max=max logit per token; topk_mass=sum of top-k softmax probabilities per token."
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="k for --metric topk_mass (sum of top-k attention probabilities per token).",
    )

    parser.add_argument(
        "--hist-output",
        type=str,
        default=str(SCRIPT_DIR / "attn_logits_hist.png"),
    )
    parser.add_argument("--hist-bins", type=int, default=400)
    parser.add_argument("--max-samples", type=int, default=2_000_000)
    parser.add_argument("--sample-per-call", type=int, default=100_000)

    parser.add_argument("--example", action="store_true")
    parser.add_argument(
        "--example-batch-idx",
        type=int,
        default=0,
        help="Batch index to snapshot when --example is set.",
    )
    parser.add_argument(
        "--example-token-idx",
        type=int,
        default=-1,
        help="Query token index to snapshot when --example is set (negative allowed).",
    )
    parser.add_argument(
        "--example-topn",
        type=int,
        default=64,
        help="How many top probabilities to show in the example plot.",
    )
    parser.add_argument(
        "--example-output",
        type=str,
        default=str(SCRIPT_DIR / "attn_example.png"),
        help="Output path for the example token distribution PNG.",
    )
    parser.add_argument("--example-full", action="store_true")
    parser.add_argument(
        "--example-full-output",
        type=str,
        default=str(SCRIPT_DIR / "attn_example_full.png"),
        help="Output path for the example token full per-key plot PNG.",
    )
    parser.add_argument(
        "--example-full-value",
        type=str,
        default="logits",
        choices=("logits", "probs"),
        help="What to plot in the example-full plot.",
    )

    return parser.parse_args()


def attn_calib_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_logits = attn_logits + causal_mask

    collector = getattr(module, "_attn_calib_collector", None)
    if collector is not None:
        collector(module, attn_logits)

    attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def register_attention_logits_collector(
    model: torch.nn.Module,
    layer_index: int,
    head_index: int,
    metric: str,
    topk: int,
    capture_example: bool,
    example_batch_idx: int,
    example_token_idx: int,
    max_samples: int,
    sample_per_call: int,
) -> tuple[list[torch.Tensor], dict[str, int], list[torch.Tensor | None]]:
    layers = getattr(model, "model").layers
    if not (0 <= layer_index < len(layers)):
        raise ValueError(f"Invalid layer index {layer_index}; model has {len(layers)} layers")

    num_heads = int(getattr(model, "config").num_attention_heads)
    if not (0 <= head_index < num_heads):
        raise ValueError(f"Invalid head index {head_index}; model has {num_heads} attention heads")

    activations: list[torch.Tensor] = []
    state = {"count": 0}
    example_logits: list[torch.Tensor | None] = [None]

    def collector(mod: nn.Module, logits: torch.Tensor) -> None:
        if int(getattr(mod, "_attn_calib_layer_index", -1)) != int(layer_index):
            return

        if capture_example and example_logits[0] is None:
            b = int(example_batch_idx)
            if 0 <= b < int(logits.shape[0]):
                q = int(example_token_idx)
                q_len = int(logits.shape[2])
                if q < 0:
                    q = q_len + q
                if 0 <= q < q_len:
                    example_logits[0] = logits[b, int(head_index), q].detach().cpu()

        remaining = int(max_samples) - int(state["count"])
        if remaining <= 0:
            return

        head_logits = logits[:, int(head_index)].detach().float()  # [B, Q, K]
        if metric == "logits":
            z = head_logits.reshape(-1)
            z = z[torch.isfinite(z)]
            if z.numel() == 0:
                return

            take = min(int(sample_per_call), remaining, int(z.numel()))
            if take <= 0:
                return
            if take < int(z.numel()):
                idx = torch.randint(0, int(z.numel()), (take,), device=z.device)
                z = z.index_select(0, idx)
            activations.append(z.cpu())
            state["count"] += int(take)
            return

        if metric == "row_max":
            row_max = head_logits.max(dim=-1).values
            z = row_max.reshape(-1)
            z = z[torch.isfinite(z)]
            if z.numel() == 0:
                return

            take = min(int(sample_per_call), remaining, int(z.numel()))
            if take <= 0:
                return
            if take < int(z.numel()):
                idx = torch.randint(0, int(z.numel()), (take,), device=z.device)
                z = z.index_select(0, idx)
            activations.append(z.cpu())
            state["count"] += int(take)
            return

        # Masking can introduce -inf; avoid -inf - -inf = nan by mapping non-finite logits to a large negative.
        head_logits = torch.nan_to_num(head_logits, neginf=-1e9, posinf=1e9)

        if metric == "avg_gap":
            if head_logits.shape[-1] < 2:
                return
            top2 = torch.topk(head_logits, k=2, dim=-1).values  # [B, Q, 2]
            z = (top2[..., 0] - top2[..., 1]).reshape(-1)
        elif metric == "topk_mass":
            k = int(topk)
            if k <= 0:
                raise ValueError("--topk must be > 0")
            k = min(k, int(head_logits.shape[-1]))
            row_max = head_logits.max(dim=-1, keepdim=True).values
            x = head_logits - row_max
            log_denom = torch.logsumexp(x, dim=-1)  # [B, Q]
            topk_vals = torch.topk(x, k=k, dim=-1).values  # [B, Q, k]
            num = torch.exp(topk_vals).sum(dim=-1)  # [B, Q]
            z = num / torch.exp(log_denom)
            z = z.reshape(-1)
        else:
            return

        z = z[torch.isfinite(z)]
        if z.numel() == 0:
            return

        take = min(int(sample_per_call), remaining, int(z.numel()))
        if take <= 0:
            return
        if take < int(z.numel()):
            idx = torch.randint(0, int(z.numel()), (take,), device=z.device)
            z = z.index_select(0, idx)
        activations.append(z.cpu())
        state["count"] += int(take)

    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        setattr(attn, "_attn_calib_layer_index", int(idx))
        setattr(attn, "_attn_calib_collector", collector)

    return activations, state, example_logits


def clear_attention_logits_collectors(model: torch.nn.Module) -> None:
    for layer in getattr(model, "model").layers:
        attn = layer.self_attn
        if hasattr(attn, "_attn_calib_collector"):
            delattr(attn, "_attn_calib_collector")
        if hasattr(attn, "_attn_calib_layer_index"):
            delattr(attn, "_attn_calib_layer_index")


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    model_path = resolve_model_path(args.model_path)
    print(
        f"[attn_calib] loading model from {model_path} (device={device}, dtype={dtype})",
        flush=True,
    )
    model = load_model(model_path, device, dtype)

    ALL_ATTENTION_FUNCTIONS["attn_calib"] = attn_calib_attention_forward
    model.model.config._attn_implementation = "attn_calib"

    activations, state, example_holder = register_attention_logits_collector(
        model,
        layer_index=int(args.layer_index),
        head_index=int(args.head_index),
        metric=str(args.metric),
        topk=int(args.topk),
        capture_example=bool(args.example),
        example_batch_idx=int(args.example_batch_idx),
        example_token_idx=int(args.example_token_idx),
        max_samples=int(args.max_samples),
        sample_per_call=int(args.sample_per_call),
    )

    cfg = Config(batch_size=args.batch_size, seq_len=args.seq_len, shuffle_buffer_size=0)
    dataloader = build_dataloader(cfg, model_path, world_size=1, rank=0)
    batch_iter = iter(dataloader)

    print(f"[attn_calib] streaming {args.batch_count} batches", flush=True)

    with torch.no_grad():
        for _ in tqdm(range(args.batch_count), desc="attn-calib", disable=False):
            batch = next(batch_iter)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

            if int(state["count"]) >= int(args.max_samples):
                break

    clear_attention_logits_collectors(model)

    if len(activations) == 0:
        raise RuntimeError("No attention logits collected (did the model run any forward passes?)")

    z = torch.cat(activations, dim=0)
    if z.numel() > int(args.max_samples):
        z = z[: int(args.max_samples)]

    print(
        f"[inspect] layer={int(args.layer_index)} head={int(args.head_index)} metric={str(args.metric)} samples={int(z.numel())}",
        flush=True,
    )
    print_activation_stats(z)

    xlabel = (
        "attention logit gap (top1 - top2)"
        if str(args.metric) == "avg_gap"
        else (
            f"top-{int(args.topk)} attention probability mass"
            if str(args.metric) == "topk_mass"
            else ("attention logit (pre-softmax)" if str(args.metric) == "logits" else "max attention logit")
        )
    )

    save_histogram_png_with_xlabel(
        z,
        Path(args.hist_output),
        bins=int(args.hist_bins),
        title=(
            f"Layer {int(args.layer_index)} head {int(args.head_index)} metric={str(args.metric)} histogram "
            f"(clipped to [p0.1%, p99.9%])"
        ),
        stats_text=format_activation_stats(z),
        xlabel=xlabel,
    )

    if bool(args.example):
        example_logits = example_holder[0]
        if example_logits is None:
            print(
                "[inspect] did not capture example logits; check --example-batch-idx/--example-token-idx",
                flush=True,
            )
        else:
            save_attention_token_distribution_png(
                example_logits,
            Path(args.example_output),
            title=(
                f"Layer {int(args.layer_index)} head {int(args.head_index)} token={int(args.example_token_idx)} "
                f"(batch={int(args.example_batch_idx)})"
            ),
            topn=int(args.example_topn),
            )
            print(f"[inspect] wrote example plot: {args.example_output}", flush=True)

            if bool(args.example_full):
                save_attention_token_full_vector_png(
                    example_logits,
                    Path(args.example_full_output),
                    title=(
                        f"Layer {int(args.layer_index)} head {int(args.head_index)} token={int(args.example_token_idx)} "
                        f"(batch={int(args.example_batch_idx)}) full"
                    ),
                    value=str(args.example_full_value),
                )
                print(f"[inspect] wrote example full plot: {args.example_full_output}", flush=True)
    print(f"[inspect] wrote histogram: {args.hist_output}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
