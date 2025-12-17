import argparse
import copy
import math
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import time
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence
from transformers.models.olmo3 import Olmo3ForCausalLM, Olmo3Model
from tqdm.auto import tqdm

from .quant import QuantLinearWithWeights
from .data import build_dataloader

DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    steps: int = 8000
    batch_size: int = 8
    seq_len: int = 1024
    accumulate_steps: int = 16
    lr: float = 5e-6
    device: str = "cuda"
    dtype: str = "bfloat16"
    base_path: str = "/workspace/.hf_home/hub/models--allenai--Olmo-3-7B-Think"
    output_dir: str = str(DIR / "checkpoints" / "student_final")
    dataset_sft: Optional[str] = "allenai/Dolci-Think-SFT-7B"
    dataset_dpo: Optional[str] = "allenai/Dolci-Think-DPO-7B"
    dataset_rl: Optional[str] = "allenai/Dolci-Think-RL-7B"
    dataset_ratio_sft: float = 0.6
    dataset_ratio_dpo: float = 0.2
    dataset_ratio_rl: float = 0.2
    shuffle_buffer_size: int = 1000
    seed: int = 0
    num_workers: int = 0
    checkpoint_interval: int = 2000
    starting_step: int = 0
    train_layers: Optional[tuple[int, ...]] = field(default=None)
    compile: bool = True


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
    parser.add_argument("--shuffle-buffer-size", type=int, default=Config.shuffle_buffer_size)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
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
        "--accumulate-steps",
        type=int,
        default=Config.accumulate_steps,
        help="Number of micro-batches to accumulate before an optimizer step.",
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=Config.compile,
        help="Use torch.compile on the student model for speed (may increase startup time).",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile even if default is on.",
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
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
        starting_step=args.starting_step,
        train_layers=train_layers,
        compile=args.compile,
        dtype=args.dtype,
        accumulate_steps=args.accumulate_steps,
    )


def checkpoint_path(step: int) -> Path:
    return DIR / "checkpoints" / f"step_{step}.pt"

def activation_calib_path(layer_index: int, group_name: str):
    act_scales_dir = DIR / "activation_calibration" / "values"
    return act_scales_dir / f"{layer_index}" / f"{group_name}.pt"

def weight_calib_path(layer_index: int, name: str):
    weight_scales_dir = DIR / "weight_calibration" / "values"
    return weight_scales_dir / f"{layer_index}" / f"{name}.pt"

def diag_scaling_path(layer_index: int, name: str):
    diag_scaling_dir = DIR / "diag_scaling" / "values"
    return diag_scaling_dir / f"{layer_index}" / f"{name}.pt"


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


def quant_max_activation(model: torch.nn.Module) -> tuple[float | None, str | None]:
    best_val: float | None = None
    best_name: str | None = None
    for module in model.modules():
        if isinstance(module, QuantLinearWithWeights) and module.last_max_act is not None:
            val = float(module.last_max_act)
            if (best_val is None) or (val > best_val):
                best_val = val
                best_name = getattr(module, "q_name", module.__class__.__name__)
    return best_val, best_name


def load_model(model_path: str, device: torch.device, dtype: torch.dtype) -> Olmo3ForCausalLM:
    # model = LlamaForCausalLM.from_pretrained(model_path, dtype=dtype)
    model = Olmo3ForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.config.use_cache = False
    model.eval()
    model = model.to(device)
    return model


def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def prepare_quant_layers(
    model: Olmo3Model,
    train_layers: tuple[int, ...],
    set_scales: bool = False
) -> None:
    for layer_index, layer in enumerate(model.layers):
        if layer_index not in train_layers:
            continue
        groups = [
            # (layer.self_attn, "q_k_v"),
            # (layer.self_attn, "o"),
            (layer.mlp, "gate_up"),
            (layer.mlp, "down"),
        ]
        for parent_module, group_name in groups:
            act_calib_payload = torch.load(activation_calib_path(layer_index, group_name), map_location="cpu")
            clip_value = act_calib_payload.get("clip_value")

            for name in [f"{prefix}_proj" for prefix in group_name.split("_")]:
                linear_module = getattr(parent_module, name)
                quant = QuantLinearWithWeights(
                    linear_module.in_features,
                    linear_module.out_features,
                )
                setattr(parent_module, name, quant)
                quant.weight = linear_module.weight
                quant.set_trainable(True)
                quant.to(linear_module.weight.device)
                quant.q_name = f"layer{layer_index}.{parent_module.__class__.__name__}.{name}"
                
                if set_scales:
                    # Set activation scale
                    quant.set_activation_scale(clip_value)
                    # Set weight scales
                    calib_weight_payload = torch.load(weight_calib_path(layer_index, name), map_location="cpu")
                    quant.set_weight_scales(calib_weight_payload.get("scales"))


def iter_layer_linears(
    layers: Iterable[torch.nn.Module],
) -> Iterator[tuple[int, torch.nn.Module, str, torch.nn.Module]]:
    """Yield each target linear module in decoder layers."""

    targets = (
        ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
        ("mlp", ("gate_proj", "up_proj", "down_proj")),
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
    model: Olmo3ForCausalLM,
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


def ensure_adamw_state_dtype(optimizer: torch.optim.AdamW, target_dtype: torch.dtype) -> None:
    """Make sure AdamW moment tensors live in the desired dtype (bf16 support)."""
    if target_dtype != torch.bfloat16:
        return

    for group in optimizer.param_groups:
        for param in group["params"]:
            if param is None or not param.requires_grad:
                continue
            state = optimizer.state[param]
            if "exp_avg" in state:
                state["exp_avg"] = state["exp_avg"].to(dtype=target_dtype)
            else:
                state["exp_avg"] = torch.zeros_like(
                    param, dtype=target_dtype, memory_format=torch.preserve_format
                )
            if "exp_avg_sq" in state:
                state["exp_avg_sq"] = state["exp_avg_sq"].to(dtype=target_dtype)
            else:
                state["exp_avg_sq"] = torch.zeros_like(
                    param, dtype=target_dtype, memory_format=torch.preserve_format
                )
            if "step" not in state:
                state["step"] = torch.zeros((), device=param.device, dtype=torch.int64)


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
    freeze_model(teacher)
    student = copy.deepcopy(teacher)
    target_layers = cfg.train_layers
    if target_layers is None:
        target_layers = tuple(range(len(student.model.layers)))
    prepare_quant_layers(
        student.model,
        target_layers,
        cfg.starting_step == 0
    )

    if cfg.compile:
        if rank == 0:
            print("[run] compiling student with torch.compile...", flush=True)
        student = torch.compile(
            student,
            fullgraph=False,
            options={"triton.cudagraphs": False},  # avoid cudagraph capture issues
        )
        if rank == 0:
            print("[run] student compile finished", flush=True)

    if distributed:
        student = DDP(student, device_ids=[rank], output_device=rank)

    # Separate LR for activation scales (log_act_s): 50x base LR
    act_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("log_act_s"):
            act_params.append(param)
        else:
            other_params.append(param)

    # beta2 = 0.95 if use_bf16 else 0.999
    # eps = 1e-10 if use_bf16 else 1e-8
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": cfg.lr},
            {"params": act_params, "lr": cfg.lr * 50.0},
        ],
        lr=cfg.lr,
        # betas=(0.9, beta2),
        # eps=eps,
        weight_decay=0.1,
        foreach=False,  # avoid multi-tensor grouping dtype issues with bf16 state casting
    )
    # ensure_adamw_state_dtype(optimizer, dtype)
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
        ensure_adamw_state_dtype(optimizer, dtype)
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
                f.write("step,kl,top1_miss,ema_miss,top1_acc,flip_penalty,median_rel_margin,flip_severity,max_act,max_act_name\n")

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

    log_interval = 12  # reuse checkpoint interval semantics

    last_kl: torch.Tensor | None = None
    last_top1_miss: torch.Tensor | None = None
    last_top1_acc: torch.Tensor | None = None
    last_flip_penalty: torch.Tensor | None = None
    last_median_rel_margin: torch.Tensor | None = None
    last_flip_severity: torch.Tensor | None = None
    last_max_act: float | None = None
    last_max_act_name: str | None = None

    for step in progress:
        optimizer.zero_grad(set_to_none=True)

        # Gradient accumulation over micro-batches
        for micro in range(cfg.accumulate_steps):
            batch = next(batch_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            with torch.no_grad():
                metrics = argmax_stability_metrics(teacher_logits, student_logits, attention_mask)
                top1_acc = metrics["top1_acc"]
                top1_miss = 1.0 - top1_acc
                flip_penalty = metrics["avg_flip_penalty"]
                median_rel_margin = metrics["median_rel_margin"]
                flip_severity = metrics["avg_flip_severity"]

            kl_loss = compute_kl_loss(student_logits, teacher_logits, attention_mask)
            best_act, best_act_name = quant_max_activation(unwrap_model(student))

            loss = kl_loss / cfg.accumulate_steps
            loss.backward()

            # Cache last micro metrics for logging outside the accumulation loop
            last_kl = kl_loss.detach()
            last_top1_miss = top1_miss.detach()
            last_top1_acc = top1_acc.detach()
            last_flip_penalty = flip_penalty.detach()
            last_median_rel_margin = median_rel_margin.detach()
            last_flip_severity = flip_severity.detach()
            last_max_act = best_act
            last_max_act_name = best_act_name

            # Drop references promptly to release graph memory
            del teacher_logits, kl_loss, student_logits, loss

        # Cast accumulated grads to configured dtype to keep optimizer state consistent
        for p in student.parameters():
            if p.grad is not None:
                p.grad.data = p.grad.data.to(dtype)

        if use_bf16:
            ensure_adamw_state_dtype(optimizer, dtype)
        optimizer.step()
        scheduler.step()

        # Fallbacks if somehow caches are None
        kl_val = last_kl.item() if last_kl is not None else 0.0
        top1_miss_val = last_top1_miss.item() if last_top1_miss is not None else 0.0
        top1_acc_val = last_top1_acc.item() if last_top1_acc is not None else 0.0
        flip_pen_val = last_flip_penalty.item() if last_flip_penalty is not None else 0.0
        median_rel_val = last_median_rel_margin.item() if last_median_rel_margin is not None else 0.0
        flip_sev_val = last_flip_severity.item() if last_flip_severity is not None else 0.0
        max_act_val = last_max_act if last_max_act is not None else 0.0
        max_act_name = last_max_act_name if last_max_act_name is not None else "n/a"

        if rank == 0 and ((step + 1) % log_interval == 0 or step == start_step):
            ema_miss = top1_miss_val if ema_miss is None else 0.9 * ema_miss + 0.1 * top1_miss_val
            with open(log_path, "a") as f:
                f.write(
                    f"{step},{kl_val:.6f},{top1_miss_val:.6f},{ema_miss:.6f},"
                    f"{top1_acc_val:.6f},{flip_pen_val:.6f},"
                    f"{median_rel_val:.6f},{flip_sev_val:.6f},"
                    f"{max_act_val:.6f},{max_act_name}\n"
                )

        progress.set_postfix_str(
            (
                f"kl={kl_val:.4f} "
                f"max_act={max_act_val:.4f}({max_act_name}) "
                f"top1_miss={top1_miss_val:.3f} ema_miss={ema_miss if ema_miss is not None else top1_miss_val:.3f} "
                f"flip_pen={flip_pen_val:.3f} "
                f"rel_median={median_rel_val:.3f} "
                f"flip_sev={flip_sev_val:.3f} "
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
