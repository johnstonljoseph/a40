import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.olmo3 import Olmo3ForCausalLM


def anneal_at_step(step: int, end: Optional[int]) -> float:
    """Cosine anneal mixing coefficient in [0,1]: 0 -> start (SiLU), 1 -> end (ReLU)."""
    if end is None or end <= 0 or step >= end:
        return 1.0
    frac = step / float(end)
    return 0.5 * (1.0 - math.cos(math.pi * frac))


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


def load_model(model_path: str, device: torch.device, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.config.use_cache = False
    model.eval()
    model = model.to(device)
    return model


def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


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


def save_checkpoint(student: nn.Module, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = student
    while hasattr(model_to_save, "_orig_mod"):
        model_to_save = model_to_save._orig_mod
    model_to_save.save_pretrained(str(output_dir))


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
    T: float = 1.0,
) -> torch.Tensor:
    s = (student_logits / T).log_softmax(dim=-1)
    t = (teacher_logits / T).log_softmax(dim=-1).exp()
    token_kl = (t * (t.log() - s)).sum(dim=-1)
    mask = attention_mask.to(dtype=token_kl.dtype)
    return (token_kl * mask).sum() / mask.sum() * (T * T)


def maybe_compile(teacher, student, rank: int) -> tuple[nn.Module, nn.Module]:
    if cfg.compile:
        if rank == 0:
            print("[run] compiling student with torch.compile...", flush=True)
        teacher = torch.compile(
            teacher,
            fullgraph=False,
            options={"triton.cudagraphs": False},  # avoid cudagraph capture issues
        )
        student = torch.compile(
            student,
            fullgraph=False,
            options={"triton.cudagraphs": False},  # avoid cudagraph capture issues
        )
        if rank == 0:
            print("[run] student compile finished", flush=True)
    (teacher, student)


def argmax_stability_metrics(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[float, float]:
    teacher_argmax = teacher_logits.argmax(dim=-1)
    student_argmax = student_logits.argmax(dim=-1)
    mask = attention_mask.to(dtype=teacher_argmax.dtype)
    correct = (teacher_argmax == student_argmax).to(dtype=mask.dtype)
    top1_acc = (correct * mask).sum() / mask.sum()
    flip_pen = ((teacher_argmax != student_argmax) * mask).sum() / mask.sum()
    return top1_acc.item(), flip_pen.item()



class WeightedRMSNorm(torch.nn.Module):
    """RMSNorm variant that assumes upstream weights already include the gamma scale.

    Given inputs x = diag(gamma) * o, this module outputs
        x / sqrt(mean((x / gamma)^2) + eps),
    which matches gamma * RMSNorm(o).
    """

    def __init__(self, base_norm: torch.nn.Module, gamma: torch.Tensor | None = None):
        super().__init__()
        if gamma is None:
            gamma = base_norm.weight.detach().clone()
        else:
            gamma = gamma.detach().clone()
        self.register_buffer("gamma", gamma)
        self.variance_epsilon = float(getattr(base_norm, "variance_epsilon", 1e-6))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.to(dtype=hidden_states.dtype, device=hidden_states.device)
        inv_gamma = torch.where(
            gamma != 0,
            1.0 / gamma,
            torch.zeros_like(gamma),
        )
        norm_input = hidden_states * inv_gamma
        variance = norm_input.pow(2).mean(dim=-1, keepdim=True)
        denom = torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states * denom
