import argparse
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from transformers import LlamaForCausalLM
from tqdm.auto import tqdm
import lm_eval
from lm_eval.models.huggingface import HFLM

from quant import QuantLinear
from data import build_dataloader


@dataclass
class Config:
    steps: int = 1
    grad_accum_steps: int = 1
    batch_size: int = 1
    seq_len: int = 16
    device: str = "cpu"
    # CPU loads are faster in float32 even if the base checkpoint ships bf16 weights.
    dtype: str = "float32"
    student_model_path: str = "~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct"
    teacher_model_path: str = "~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct"
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
    eval_tasks: List[str] = field(default_factory=lambda: ["hellaswag", "piqa", "winogrande"])


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=Config.batch_size)
    args = parser.parse_args()
    return Config(steps=args.steps, batch_size=args.batch_size)


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


def load_model(model_path: str, device: torch.device, dtype: torch.dtype) -> LlamaForCausalLM:
    print(f"  -> from_pretrained({model_path})", flush=True)
    model = LlamaForCausalLM.from_pretrained(model_path, dtype=dtype)
    print("  -> configuring", flush=True)
    model.config.use_cache = False
    model.eval()
    print(f"  -> moving to {device}", flush=True)
    model = model.to(device)
    print("  -> model ready", flush=True)
    return model


def freeze(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _iter_linear_modules(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            yield module, name, child
        else:
            yield from _iter_linear_modules(child)


def swap_linear_with_quant(module: torch.nn.Module) -> None:
    linears = list(_iter_linear_modules(module))
    for parent, name, child in tqdm(linears, desc="swap linears", leave=False):
        quant = QuantLinear(child.in_features, child.out_features)
        # Reuse the existing weight parameter to avoid an extra copy.
        quant.weight = child.weight
        quant.initialize(child, show_progress=True, desc=f"scales:{name}")
        setattr(parent, name, quant)


def select_quant_params(model: LlamaForCausalLM):
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.weight.requires_grad = True
            module.log_weight_s.requires_grad = True
            module.log_act_s.requires_grad = True


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
    lm = HFLM(pretrained=model, batch_size=cfg.batch_size)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=cfg.eval_tasks,
        batch_size=cfg.batch_size,
    )
    metrics = {}
    for task, task_results in results["results"].items():
        acc = task_results.get("acc,none") or task_results.get("acc")
        if acc is not None:
            metrics[task] = acc
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
    print("  -> freezing student", flush=True)
    freeze(student)
    print("  -> swapping Linear -> QuantLinear", flush=True)
    swap_linear_with_quant(student)
    print("  -> enabling grads on quant params", flush=True)
    select_quant_params(student)

    print("Loading teacher model...", flush=True)
    teacher = load_model(teacher_path, device, dtype)
    freeze(teacher)

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, Tmax=cfg.steps, eta_min=cfg.lr * 0.1
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

        optimizer.step()
        scheduler.step()

        if step % cfg.log_interval == 0 or step == cfg.steps - 1:
            progress.set_postfix(kl_loss=f"{kl_total:.6f}")

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

