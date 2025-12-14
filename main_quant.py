import argparse
import copy
import math
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence
from transformers import LlamaForCausalLM, LlamaModel
from tqdm.auto import tqdm

from .quant import QuantLinear
from .data import build_dataloader

DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    steps: int = 8000
    batch_size: int = 4
    seq_len: int = 512
    lr: float = 5e-6
    device: str = "cuda"
    dtype: str = "bfloat16"
    base_path: str = "/workspace/.hf_home/hub/models--allenai--Llama-3.1-Tulu-3.1-8B"
    # models--meta-llama--Llama-3.2-1B-Instruct"
    # base_path: str = "/Users/joseph/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct"
    dataset_a: str = "allenai/tulu-3-sft-mixture"
    dataset_b: str = "mlfoundations/dclm-baseline-1.0"
    dataset_ratio_a: float = 0.5
    dataset_split: str = "train"
    shuffle_buffer_size: int = 10_000
    seed: int = 0
    num_workers: int = 1
    checkpoint_interval: int = 2000
    starting_step: int = 0
    train_layers: Optional[tuple[int, ...]] = field(default=None)
    penalty_scale: float = 0
    margin_weight: float = 0.0
    align_weight: float = 32.0


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=Config.batch_size)
    parser.add_argument(
        "--seq-len",
        dest="seq_len",
        type=int,
        default=Config.seq_len,
        help="Sequence length for packed training batches",
    )
    parser.add_argument(
        "--train-layers",
        type=str,
        default="all",
        help="Comma-separated decoder layer indices to finetune or 'all' (default)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=Config.checkpoint_interval,
        help="Save a checkpoint every N steps (0 to disable)",
    )
    parser.add_argument(
        "--starting-step",
        type=int,
        default=Config.starting_step,
        help="Step number to begin training from (0 to start fresh).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=Config.device,
        help="Device to run distillation on (e.g., cuda:x or cpu).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=Config.dtype,
        help="Torch dtype for loading models (e.g., float32, bfloat16).",
    )
    parser.add_argument(
        "--weight-penalty-scale",
        type=float,
        default=Config.penalty_scale,
        help="Multiplier for quant weight penalty loss.",
    )
    parser.add_argument(
        "--margin-weight",
        type=float,
        default=Config.margin_weight,
        help="Weight for margin component in anchor loss.",
    )
    parser.add_argument(
        "--align-weight",
        type=float,
        default=Config.align_weight,
        help="Weight for align (teacher top-1) component in anchor loss.",
    )
    args = parser.parse_args()
    raw_train_layers = args.train_layers.strip()
    train_layers = (
        None
        if raw_train_layers.lower() == "all"
        else tuple(int(tok) for tok in raw_train_layers.split(",") if tok.strip())
    )
    return Config(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        train_layers=train_layers,
        checkpoint_interval=args.checkpoint_interval,
        starting_step=args.starting_step,
        device=args.device,
        dtype=args.dtype,
        penalty_scale=args.weight_penalty_scale,
        margin_weight=args.margin_weight,
        align_weight=args.align_weight,
    )


def checkpoint_path(step: int) -> Path:
    return DIR / "checkpoints" / f"step_{step}.pt"

def activation_calib_path(layer_index: int, group_name: str):
    act_scales_dir = DIR / "activation_calibration" / "values"
    return act_scales_dir / f"{layer_index}" / f"{group_name}.pt"

def weight_calib_path(layer_index: int, name: str):
    weight_scales_dir = DIR / "weight_calibration" / "values"
    return weight_scales_dir / f"{layer_index}" / f"{name}.pt"


def resolve_model_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    snapshots = path / "snapshots"
    if snapshots.exists():
        latest = sorted(snapshots.iterdir())
        if not latest:
            raise FileNotFoundError(f"No snapshots found under {snapshots}")
        return str(latest[-1])

    if path.is_dir():
        return str(path)

    raise FileNotFoundError(
        f"Model path {path} not found. Expected a HF snapshot dir or concrete model folder."
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def spikes_penalty(model: torch.nn.Module) -> torch.Tensor:
    """Aggregate quant penalties; always returns a tensor on model device."""
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    total = torch.tensor(0.0, device=device)
    for module in model.modules():
        if isinstance(module, QuantLinear) and module.last_penalty is not None:
            total += module.last_penalty
    return total


def quant_max_activation(model: torch.nn.Module) -> tuple[float | None, str | None]:
    best_val: float | None = None
    best_name: str | None = None
    for module in model.modules():
        if isinstance(module, QuantLinear) and module.last_max_act is not None:
            val = float(module.last_max_act)
            if (best_val is None) or (val > best_val):
                best_val = val
                best_name = getattr(module, "q_name", module.__class__.__name__)
    return best_val, best_name


def load_model(model_path: str, device: torch.device, dtype: torch.dtype) -> LlamaForCausalLM:
    model = LlamaForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.config.use_cache = False
    model.eval()
    model = model.to(device)
    return model


def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def prepare_quant_layers(
    llama_model: LlamaModel,
    train_layers: tuple[int, ...],
    set_scales: bool = False
) -> None:
    for layer_index, layer in enumerate(llama_model.layers):
        if layer_index not in train_layers:
            continue
        groups = [
            (layer.self_attn, "q_k_v"),
            (layer.self_attn, "o"),
            (layer.mlp, "gate_up"),
        ]
        for parent_module, group_name in groups:
            act_calib_payload = torch.load(activation_calib_path(layer_index, group_name), map_location="cpu")
            clip_value = act_calib_payload.get("clip_value")

            for name in [f"{prefix}_proj" for prefix in group_name.split("_")]:
                linear_module = getattr(parent_module, name)
                quant = QuantLinear(
                    linear_module.in_features,
                    linear_module.out_features,
                )
                setattr(parent_module, name, quant)
                quant.weight = linear_module.weight
                quant.set_trainable(True)
                quant.to(linear_module.weight.device)
                quant.q_name = f"layer{layer_index}.{parent_module.__class__.__name__}.{name}"
                
                if set_scales:
                    quant.set_activation_scale(clip_value)
                    calib_weight_payload = torch.load(weight_calib_path(layer_index, name), map_location="cpu")
                    quant.set_weight_scales(calib_weight_payload.get("scales"))


def iter_layer_linears(
    layers: Iterable[torch.nn.Module],
) -> Iterator[tuple[int, torch.nn.Module, str, torch.nn.Module]]:
    """Yield each target linear module in decoder layers."""

    targets = (
        ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
        ("mlp", ("gate_proj", "up_proj")),
    )

    for layer_index, layer in enumerate(layers):
        for parent_name, child_names in targets:
            parent = getattr(layer, parent_name, None)
            if parent is None:
                continue
            for name in child_names:
                child = getattr(parent, name, None)
                if child is None:
                    continue
                yield layer_index, parent, name, child


def load_checkpoint(
    checkpoint_path: str,
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: Optional[torch.device] = None,
) -> None:
    """Load checkpoint into model (must already have QuantLinear modules)."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])


def compute_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    student_log_probs = student_logits.log_softmax(dim=-1)
    teacher_log_probs = teacher_logits.log_softmax(dim=-1)
    teacher_probs = teacher_log_probs.exp()
    kl = teacher_probs * (teacher_log_probs - student_log_probs)
    token_kl = kl.sum(dim=-1)
    mask = attention_mask.float()
    return (token_kl * mask).sum() / mask.sum()


def compute_token_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().view(-1).float()
    vocab_size = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    )
    mask_sum = shift_mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, device=logits.device)
    return (loss * shift_mask).sum() / mask_sum


def compute_anchor_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    align_weight: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    mask = attention_mask.float()
    temp = 4.0
    teacher_idx = teacher_logits.argmax(dim=-1)
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    align_loss = -student_log_probs.gather(-1, teacher_idx.unsqueeze(-1)).squeeze(-1)

    denom = mask.sum()
    align = (align_loss * mask).sum() / denom

    grad_align = torch.autograd.grad(align, student_logits, retain_graph=True)[0]
    grad_align_top = (grad_align.gather(-1, teacher_idx.unsqueeze(-1)).squeeze(-1) * mask).sum() / denom
    grad_align_norm = torch.sqrt((grad_align.pow(2).sum(dim=-1) * mask).sum() / denom)

    return (
        (align_weight * align_loss * mask).sum() / denom,
        align,
        grad_align_top,
        grad_align_norm,
    )


def argmax_stability_metrics(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    tau: float = 0.7,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Alignment diagnostics on masked tokens."""
    token_mask = attention_mask.bool()
    num_tokens = token_mask.sum()
    if num_tokens == 0:
        zero = torch.tensor(0.0, device=teacher_logits.device)
        return {
            "top1_acc": zero,
            "avg_flip_penalty": zero,
            "median_rel_margin": zero,
            "avg_flip_severity": zero,
        }

    t_flat = teacher_logits[token_mask]  # [N, V]
    s_flat = student_logits[token_mask]

    top2 = t_flat.topk(2, dim=-1)
    idx_top1 = top2.indices[:, 0]
    idx_top2 = top2.indices[:, 1]

    zi = t_flat.gather(-1, idx_top1[:, None]).squeeze(-1)
    zj = t_flat.gather(-1, idx_top2[:, None]).squeeze(-1)
    m = zi - zj

    zhi = s_flat.gather(-1, idx_top1[:, None]).squeeze(-1)
    zhj = s_flat.gather(-1, idx_top2[:, None]).squeeze(-1)
    mhat = zhi - zhj

    student_top1 = s_flat.argmax(dim=-1)
    top1_correct = student_top1 == idx_top1
    top1_acc = top1_correct.float().mean()

    flip_penalty = (~top1_correct) * torch.tanh(m / tau)
    avg_flip_penalty = flip_penalty.mean()

    rel_margin = mhat / (m + eps)
    rel_margin_aligned = rel_margin[top1_correct]
    if rel_margin_aligned.numel() == 0:
        median_rel_margin = torch.tensor(0.0, device=teacher_logits.device)
    else:
        median_rel_margin = rel_margin_aligned.median()

    k = student_top1
    zhk = s_flat.gather(-1, k[:, None]).squeeze(-1)
    flip_severity = torch.clamp(zhk - zhi, min=0.0)
    avg_flip_severity = flip_severity.mean()

    return {
        "top1_acc": top1_acc,
        "avg_flip_penalty": avg_flip_penalty,
        "median_rel_margin": median_rel_margin,
        "avg_flip_severity": avg_flip_severity,
    }


def run(cfg: Config) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA-capable devices.")
        torch.cuda.set_device(rank)
        backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl")
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

    device = torch.device(f"cuda:{rank}" if distributed else cfg.device)
    dtype = getattr(torch, cfg.dtype)
    use_bf16 = dtype == torch.bfloat16

    if rank == 0:
        print("[run] resolving model path...", flush=True)
    base_path = resolve_model_path(cfg.base_path)

    if rank == 0:
        print("[run] loading teacher model...", flush=True)
    teacher = load_model(base_path, device, dtype)

    if rank == 0:
        print("[run] loading student model...", flush=True)
    student = copy.deepcopy(teacher)
    freeze_model(teacher)
    # Freeze only embedding and lm_head, train everything else
    student.model.embed_tokens.requires_grad_(False)
    student.lm_head.requires_grad_(False)
    target_layers = cfg.train_layers
    if target_layers is None:
        target_layers = tuple(range(len(student.model.layers)))
    prepare_quant_layers(
        student.model,
        target_layers,
        cfg.starting_step == 0
    )

    # Enable activation checkpointing to cut memory in half for long sequences
    student.gradient_checkpointing_enable(use_reentrant=False)

    if distributed:
        student = DDP(student, device_ids=[rank], output_device=rank)

    beta2 = 0.95 if use_bf16 else 0.999
    eps = 1e-10 if use_bf16 else 1e-8
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.lr,
        betas=(0.9, beta2),
        eps=eps,
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps, eta_min=cfg.lr * 0.1
    )

    start_step = cfg.starting_step
    if start_step > 0:
        ckpt_path = checkpoint_path(start_step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint for step {start_step} not found at {ckpt_path}")
        load_checkpoint(
            str(ckpt_path),
            unwrap_model(student),
            optimizer,
            scheduler,
            map_location=device,
        )
    if rank == 0:
        print(f"[run] resumed from step {start_step}")

    batch_iter = iter(build_dataloader(
        cfg,
        base_path,
        world_size,
        rank,
    ))

    ema_miss: float | None = None
    log_path = DIR / "logs" / "top1_miss.csv"
    if rank == 0:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists():
            with open(log_path, "w") as f:
                f.write("step,top1_miss,ema_miss,top1_acc,flip_penalty,median_rel_margin,flip_severity\n")

    progress = tqdm(
        range(start_step, cfg.steps),
        initial=start_step,
        total=cfg.steps,
        desc="distill",
        bar_format="{desc}: |{bar}| {n_fmt}/{total_fmt} {postfix}",
        dynamic_ncols=True,
        ncols=140,
        disable=(rank != 0),
    )

    for step in progress:
        optimizer.zero_grad()

        batch = next(batch_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

        # with torch.no_grad():
        #     (
        #         anchor_loss,
        #         align_loss,
        #         grad_align_top,
        #         grad_align_norm,
        #     ) = compute_anchor_loss(
        #         student_logits,
        #         teacher_logits,
        #         attention_mask,
        #         align_weight=cfg.align_weight,
        #     )

            metrics = argmax_stability_metrics(teacher_logits, student_logits, attention_mask)
            top1_acc = metrics["top1_acc"]
            top1_miss = 1.0 - top1_acc
            flip_penalty = metrics["avg_flip_penalty"]
            median_rel_margin = metrics["median_rel_margin"]
            flip_severity = metrics["avg_flip_severity"]

        # penalty_term = torch.tensor(0.0, device=device)
        loss = compute_kl_loss(student_logits, teacher_logits, attention_mask)

        best_act, best_act_name = quant_max_activation(unwrap_model(student))

        loss.backward()

        # (teacher_logits - student_logits)

        # Overall gradient norms
        with torch.no_grad():
            logits_grad_norm = (
                student_logits.grad.norm().detach()
                if student_logits.grad is not None
                else torch.tensor(0.0, device=device)
            )
            param_sq_sum = torch.zeros(1, device=device)
            for p in student.parameters():
                if p.grad is not None:
                    param_sq_sum += p.grad.pow(2).sum()
            param_grad_norm = param_sq_sum.sqrt()

        optimizer.step()
        scheduler.step()

        if rank == 0:
            # Update EMA of top1_miss
            ema_miss = top1_miss.item() if ema_miss is None else 0.9 * ema_miss + 0.1 * top1_miss.item()
            # Append to CSV log (rank0)
            with open(log_path, "a") as f:
                f.write(
                    f"{step},{top1_miss.item():.6f},{ema_miss:.6f},"
                    f"{top1_acc.item():.6f},{flip_penalty.item():.6f},"
                    f"{median_rel_margin.item():.6f},{flip_severity.item():.6f}\n"
                )
            max_act_val = best_act if best_act is not None else 0.0
            max_act_name = best_act_name if best_act_name is not None else "n/a"
            progress.set_postfix_str(
                (
                    # f"align={align_loss.item():.4f} "
                    f"KL={loss.item():.6f} "
                    f"max_act={max_act_val:.4f}({max_act_name}) "
                    f"top1_miss={top1_miss.item():.3f} ema_miss={ema_miss:.3f} "
                    f"flip_pen={flip_penalty.item():.3f} rel_median={median_rel_margin.item():.3f} "
                    f"flip_sev={flip_severity.item():.3f} "
                    f"logit_gn={logits_grad_norm.item():.4f} param_gn={param_grad_norm.item():.4f} "
                )
            )

        if cfg.checkpoint_interval > 0 and (step + 1) % cfg.checkpoint_interval == 0:
            if rank == 0:
                print(f"Saving checkpoint for step {step + 1}")
                torch.save(
                    {
                        "model": unwrap_model(student).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    checkpoint_path(step + 1)
                )
            if distributed:
                dist.barrier()

    if distributed:
        dist.destroy_process_group()


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
