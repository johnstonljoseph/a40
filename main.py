import argparse
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence
from transformers import LlamaForCausalLM
from tqdm.auto import tqdm

from .eval import evaluate_model, extract_basic_metrics
from .quant import QuantLinear
from .data import build_dataloader


class ConfigDefaults:
    eval_tasks: tuple[str, ...] = ("gsm8k", "truthfulqa_mc1", "ifeval")


@dataclass
class Config:
    steps: int = 1
    grad_accum_steps: int = 1
    batch_size: int = 1
    seq_len: int = 16
    device: str = "cuda"
    # CPU loads are faster in float32 even if the base checkpoint ships bf16 weights.
    dtype: str = "float32"
    student_model_path: str = "/workspace/.hf_home/hub/models--meta-llama--Llama-3.2-1B-Instruct"
    teacher_model_path: str = "/workspace/.hf_home/hub/models--meta-llama--Llama-3.2-1B-Instruct"
    lr: float = 0.1
    dataset_a: str = "allenai/tulu-3-sft-mixture"
    dataset_b: str = "mlfoundations/dclm-baseline-1.0"
    dataset_split: str = "train"
    dataset_ratio_a: float = 0.75
    shuffle_buffer_size: int = 10_000
    seed: int = 0
    num_workers: int = 1
    prefetch_factor: int = 2
    log_interval: int = 10
    checkpoint_interval: int = 500
    eval_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    resume_from: str = ""  # path to checkpoint.pt to resume from
    eval_tasks: List[str] = field(default_factory=lambda: list(ConfigDefaults.eval_tasks))
    train_layers: tuple[int, ...] = field(default_factory=tuple)  # decoder layer indices to update
    eval_limit: Optional[float] = None  # fraction of eval dataset per task


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=Config.batch_size)
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
        "--eval-interval",
        type=int,
        default=Config.eval_interval,
        help="Run evaluation every N steps (0 to disable)",
    )
    parser.add_argument(
        "--eval-tasks",
        type=str,
        default=",".join(ConfigDefaults.eval_tasks),
        help="Comma-separated lm_eval task names (must be non-script datasets)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=Config.log_interval,
        help="Update progress bar metrics every N steps",
    )
    parser.add_argument(
        "--seq-len",
        dest="seq_len",
        type=int,
        default=Config.seq_len,
        help="Sequence length for packed training batches",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=Config.resume_from,
        help="Path to checkpoint.pt to resume from",
    )
    parser.add_argument(
        "--eval-limit",
        dest="eval_limit",
        type=float,
        default=None,
        help="Optional fraction of eval examples per task (e.g., 0.01)",
    )
    args = parser.parse_args()
    raw_train_layers = args.train_layers.strip()
    if not raw_train_layers:
        raise ValueError("--train-layers requires at least one integer index")
    try:
        train_layers = tuple(int(tok) for tok in raw_train_layers.split(",") if tok.strip())
    except ValueError as exc:
        raise ValueError("--train-layers must be a comma-separated list of integers") from exc
    if not train_layers:
        raise ValueError("--train-layers requires at least one integer index")
    eval_tasks = tuple(
        task.strip()
        for task in args.eval_tasks.split(",")
        if task.strip()
    ) or tuple(Config.eval_tasks)

    return Config(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        train_layers=train_layers,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        eval_tasks=list(eval_tasks),
        resume_from=args.resume_from,
        eval_limit=args.eval_limit,
    )


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


def load_model(model_path: str, device: torch.device, dtype: torch.dtype) -> LlamaForCausalLM:
    model = LlamaForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.config.use_cache = False
    model.eval()
    model = model.to(device)
    return model


def freeze(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _iter_linear_modules(module: torch.nn.Module, prefix: str = ""):
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear):
            yield module, name, child
        else:
            yield from _iter_linear_modules(child, child_prefix)


def _iter_layer_linears(layers: Sequence[torch.nn.Module]):
    for layer_index, layer in enumerate(layers):
        for parent, name, child in _iter_linear_modules(layer):
            yield layer_index, parent, name, child


def swap_linear_with_quant(
    module: torch.nn.Module,
    train_layers: tuple[int, ...],
    *,
    batch_size: int,
    seq_len: int,
) -> None:
    for layer_index, parent, name, child in _iter_layer_linears(module.model.layers):
        if layer_index not in train_layers:
            continue
        quant = QuantLinear(
            child.in_features,
            child.out_features,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        quant.weight = child.weight
        quant.initialize(child, show_progress=True, desc=f"scales:{name}")
        quant.set_trainable(True)
        setattr(parent, name, quant)


def save_checkpoint(
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    cfg: Config,
) -> None:
    checkpoint_path = Path(cfg.checkpoint_dir) / f"step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        checkpoint_path / "checkpoint.pt",
    )
    print(f"Checkpoint saved at step {step} to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> int:
    """Load checkpoint into model (must already have QuantLinear modules). Returns step."""
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"]


def evaluate(model: LlamaForCausalLM, cfg: Config) -> dict:
    results = evaluate_model(
        model,
        tasks=cfg.eval_tasks,
        batch_size=cfg.batch_size,
        limit=cfg.eval_limit,
    )
    metrics = extract_basic_metrics(results.get("results", {}))
    for task, acc in metrics.items():
        print(f"  {task}: {acc:.4f}")
    return metrics


def run(cfg: Config) -> None:
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)

    print("Resolving model paths...", flush=True)
    student_path = resolve_model_path(cfg.student_model_path)
    teacher_path = resolve_model_path(cfg.teacher_model_path)

    print("Loading student model...", flush=True)
    student = load_model(student_path, device, dtype)
    freeze(student)
    swap_linear_with_quant(
        student,
        cfg.train_layers,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
    )

    print("Loading teacher model...", flush=True)
    teacher = load_model(teacher_path, device, dtype)
    freeze(teacher)

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps, eta_min=cfg.lr * 0.1
    )

    start_step = 0
    if cfg.resume_from:
        start_step = load_checkpoint(cfg.resume_from, student, optimizer, scheduler)
        print(f"Resumed from step {start_step}")

    print("Building dataloader...", flush=True)
    batch_iter = iter(build_dataloader(cfg, student_path))

    progress = tqdm(
        range(start_step, cfg.steps),
        initial=start_step,
        total=cfg.steps,
        desc="distill",
    )

    for step in progress:
        print(f"[step {step + 1}/{cfg.steps}] starting", flush=True)
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

        if step % cfg.log_interval == 0 or step == cfg.steps - 1:
            progress.set_postfix(
                kl_loss=f"{kl_total:.6f}",
                student_ppl=f"{student_ppl:.2f}",
                teacher_ppl=f"{teacher_ppl:.2f}",
            )

        if cfg.checkpoint_interval > 0 and (step + 1) % cfg.checkpoint_interval == 0:
            save_checkpoint(student, optimizer, scheduler, step + 1, cfg)

        if cfg.eval_interval > 0 and (step + 1) % cfg.eval_interval == 0:
            print(f"Evaluating at step {step + 1}...")
            evaluate(student, cfg)


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()

