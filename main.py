import argparse
import copy
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
from transformers import LlamaForCausalLM
from tqdm.auto import tqdm

from .quant import QuantLinear
from .data import build_dataloader

DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    steps: int = 10
    grad_accum_steps: int = 1
    batch_size: int = 4
    seq_len: int = 256
    lr: float = 5e-6
    device: str = "cpu"
    dtype: str = "float32"
    # base_path: str = "/workspace/.hf_home/hub/models--meta-llama--Llama-3.2-1B-Instruct"
    base_path: str = "/Users/joseph/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct"
    dataset_a: str = "allenai/tulu-3-sft-mixture"
    dataset_b: str = "mlfoundations/dclm-baseline-1.0"
    dataset_ratio_a: float = 0.75
    dataset_split: str = "train"
    shuffle_buffer_size: int = 10_000
    seed: int = 0
    num_workers: int = 1
    checkpoint_interval: int = 40
    starting_step: int = 0
    train_layers: tuple[int, ...] = field(default_factory=tuple)  # Decoder layer indices to update.


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
        required=True,
        help="Comma-separated decoder layer indices to finetune",
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
        help="Device to run distillation on (e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=Config.dtype,
        help="Torch dtype for loading models (e.g., float32, bfloat16).",
    )
    args = parser.parse_args()
    raw_train_layers = args.train_layers.strip()
    train_layers = tuple(int(tok) for tok in raw_train_layers.split(",") if tok.strip())
    return Config(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        train_layers=train_layers,
        checkpoint_interval=args.checkpoint_interval,
        starting_step=args.starting_step,
        device=args.device,
        dtype=args.dtype,
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


def load_model(model_path: str, device: torch.device, dtype: torch.dtype) -> LlamaForCausalLM:
    model = LlamaForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.config.use_cache = False
    model.eval()
    model = model.to(device)
    return model


def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def iter_linear_modules(module: torch.nn.Module, prefix: str = ""):
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear):
            yield module, name, child
        else:
            yield from iter_linear_modules(child, child_prefix)


def iter_layer_linears(layers: Sequence[torch.nn.Module]):
    for layer_index, layer in enumerate(layers):
        for parent, name, child in iter_linear_modules(layer):
            yield layer_index, parent, name, child


def hydrate_with_quant(
    parent_module: torch.nn.Module,
    train_layers: tuple[int, ...],
    hydrate_fresh_non_weight_params: bool = False
) -> None:
    for layer_index, layer in enumerate(parent_module.model.layers):
        if layer_index not in train_layers:
            continue
        groups = [
            (layer.self_attn, "q_k_v"),
            (layer.self_attn, "o"),
            (layer.mlp, "gate_up"),
            (layer.mlp, "down"),
        ]
        for parent_module, group_name in groups:
            act_calib_payload = torch.load(activation_calib_path(layer_index, group_name), map_location="cpu")
            clip_value = act_calib_payload.get("clip_value")

            for name in [f"{tok}_proj" for tok in group_name.split("_")]:
                linear_module = getattr(parent_module, name)
                quant = QuantLinear(
                    linear_module.in_features,
                    linear_module.out_features,
                )
                setattr(parent_module, name, quant)
                quant.weight = linear_module.weight
                quant.to(linear_module.weight.device)
                
                if hydrate_fresh_non_weight_params:
                    quant.set_activation_scale(clip_value)
                    calib_weight_payload = torch.load(weight_calib_path(layer_index, name), map_location="cpu")
                    quant.set_weight_scales(calib_weight_payload.get("scales"))


def load_checkpoint(
    checkpoint_path: str,
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Load checkpoint into model (must already have QuantLinear modules)."""
    ckpt = torch.load(checkpoint_path, weights_only=False)
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


def run(cfg: Config) -> None:
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)

    print("[run] resolving model path...", flush=True)
    base_path = resolve_model_path(cfg.base_path)

    print("[run] loading teacher model...", flush=True)
    teacher = load_model(base_path, device, dtype)
    freeze_model(teacher)

    print("[run] loading student model...", flush=True)
    student = copy.deepcopy(teacher)
    hydrate_with_quant(
        student,
        cfg.train_layers,
        cfg.starting_step == 0
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps, eta_min=cfg.lr * 0.1
    )

    start_step = cfg.starting_step
    if start_step > 0:
        ckpt_path = checkpoint_path(start_step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint for step {start_step} not found at {ckpt_path}")
        load_checkpoint(str(ckpt_path), student, optimizer, scheduler)
    print(f"[run] resumed from step {start_step}")

    batch_iter = iter(build_dataloader(cfg, base_path))

    progress = tqdm(
        range(start_step, cfg.steps),
        initial=start_step,
        total=cfg.steps,
        desc="distill",
        bar_format="{desc}: |{bar}| {n_fmt}/{total_fmt}{postfix}",
    )

    for step in progress:
        optimizer.zero_grad()

        kl_total = 0.0
        student_ce_total = 0.0
        teacher_ce_total = 0.0

        for _ in range(cfg.grad_accum_steps):
            batch = next(batch_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            kl_loss = compute_kl_loss(student_logits, teacher_logits, attention_mask)
            kl_loss /= cfg.grad_accum_steps
            kl_loss.backward()
            kl_total += kl_loss.item()
            student_ce = compute_token_cross_entropy(
                student_logits.detach(), input_ids, attention_mask
            )
            teacher_ce = compute_token_cross_entropy(
                teacher_logits, input_ids, attention_mask
            )
            student_ce_total += student_ce.item()
            teacher_ce_total += teacher_ce.item()

        optimizer.step()
        scheduler.step()

        avg_student_ce = student_ce_total / cfg.grad_accum_steps
        avg_teacher_ce = teacher_ce_total / cfg.grad_accum_steps
        student_ppl = math.exp(avg_student_ce)
        teacher_ppl = math.exp(avg_teacher_ce)

        progress.set_postfix_str(
            f"[kl_loss={kl_total:9.6f}, student_ppl={student_ppl:7.2f}, teacher_ppl={teacher_ppl:7.2f}]"
        )

        if cfg.checkpoint_interval > 0 and (step + 1) % cfg.checkpoint_interval == 0:
            print(f"Saving checkpoint for step {step + 1}")
            torch.save(
                {
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                checkpoint_path(step + 1)
            )
            

def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()

