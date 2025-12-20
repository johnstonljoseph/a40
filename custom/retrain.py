import copy
import os
import select
import sys
from datetime import timedelta
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers.models.olmo3 import Olmo3Model, Olmo3ForCausalLM
from a40.quant import QuantLinearWithScales, QuantLinearWithWeights
from a40.custom_model import MyOlmo3ForCausalLM  # registers custom arch

from ..data import build_dataloader
from ..utils import (
    compute_kl_loss,
    freeze_model,
    load_model,
    resolve_model_path,
    argmax_stability_metrics
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DIR = Path(__file__).resolve().parent

@dataclass
class Config:
    batch_size: int = 8
    seq_len: int = 1024
    accumulate_steps: int = 16
    lr: float = 2e-6
    device: str = "cuda"
    dtype: str = "bfloat16"
    base_path: str = "/workspace/.hf_home/hub/models--allenai--Olmo-3-7B-Think"
    output_dir: str = str(DIR.parent / "checkpoints" / "student_final")
    dataset_sft: Optional[str] = "allenai/Dolci-Think-SFT-7B"
    dataset_dpo: Optional[str] = "allenai/Dolci-Think-DPO-7B"
    dataset_rl: Optional[str] = "allenai/Dolci-Think-RL-7B"
    dataset_ratio_sft: float = 0.6
    dataset_ratio_dpo: float = 0.2
    dataset_ratio_rl: float = 0.2
    shuffle_buffer_size: int = 10000
    seed: int = 43
    num_workers: int = 1
    compile: bool = True
    kl_weight: float = 1.0
    weight_decay: float = 0.05
    log_act_s_lr_mult: float = 10.0
    log_weight_s_lr_mult: float = 1.0
    vocab_bias_lr_mult: float = 1.0
    down_proj_bias_lr_mult: float = 1.0


def weight_calib_path(layer_index: int, name: str):
    weight_scales_dir = DIR.parent / "weight_calibration" / "values"
    return weight_scales_dir / f"{layer_index}" / f"{name}.pt"


def prepare_layers(
    model: MyOlmo3ForCausalLM,
    lr: float,
    log_act_s_lr_mult: float,
    log_weight_s_lr_mult: float,
    vocab_bias_lr_mult: float,
    down_proj_bias_lr_mult: float,
) -> list[dict]:
    layer_param_groups: list[dict] = []
    for layer_index, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        names = ["gate_proj", "up_proj", "down_proj"]
        for name in names:
            linear = getattr(mlp, name)
            linear.set_trainable(True)

            layer_param_groups.append({"params": [linear.weight], "lr": lr, "lr_mult": 1.0})
            layer_param_groups.append(
                {"params": [linear.log_act_s], "lr": lr * log_act_s_lr_mult, "lr_mult": log_act_s_lr_mult}
            )
            layer_param_groups.append(
                {"params": [linear.log_weight_s], "lr": lr * log_weight_s_lr_mult, "lr_mult": log_weight_s_lr_mult}
            )
            if getattr(linear, "bias", None) is not None:
                layer_param_groups.append(
                    {
                        "params": [linear.bias],
                        "lr": lr * down_proj_bias_lr_mult,
                        "lr_mult": down_proj_bias_lr_mult,
                        "weight_decay": 0.0,
                    }
                )
            linear.q_name = f"layer{layer_index}.{mlp.__class__.__name__}.{name}"

    head_bias = getattr(model.lm_head, "bias", None)
    if head_bias is not None:
        head_bias.requires_grad_(True)
        layer_param_groups.append(
            {
                "params": [head_bias],
                "lr": lr * vocab_bias_lr_mult,
                "lr_mult": vocab_bias_lr_mult,
                "weight_decay": 0.0,
            }
        )

    return layer_param_groups


def maybe_update_lr_or_checkpoint(
    cfg,
    *,
    distributed: bool,
    device: torch.device,
    rank: int,
) -> bool:
    """Non-blocking stdin poll to update base LR or request checkpoint; broadcasts in DDP."""
    new_base_lr: float | None = None
    checkpoint_requested = False
    if rank == 0:
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
        except (OSError, ValueError):
            readable = []
        if readable:
            line = sys.stdin.readline()
            if line:
                stripped = line.strip().lower()
                if stripped == "checkpoint":
                    checkpoint_requested = True
                else:
                    try:
                        candidate = float(stripped)
                        if candidate <= 0:
                            print(f"[lr] expected positive value, got '{stripped}'", flush=True)
                        else:
                            new_base_lr = candidate
                    except ValueError:
                        print(
                            f"[lr] invalid input: '{stripped}' (enter float lr or 'checkpoint')",
                            flush=True,
                        )
    # broadcast candidate lr and exit flag so all ranks stay in sync
    if distributed:
        tensor = torch.tensor(
            [
                new_base_lr if new_base_lr is not None else -1.0,
                1.0 if checkpoint_requested else 0.0,
            ],
            device=device,
        )
        dist.broadcast(tensor, src=0)
        broadcasted_lr = float(tensor[0].item())
        broadcasted_exit = bool(tensor[1].item() > 0.5)
        if broadcasted_lr > 0:
            new_base_lr = broadcasted_lr
        else:
            new_base_lr = None
        checkpoint_requested = broadcasted_exit
    if new_base_lr is not None:
        cfg.lr = new_base_lr
        if rank == 0:
            print(f"[lr] updated base lr to {new_base_lr:.2e}", flush=True)
    return checkpoint_requested


def save_checkpoint(model, cfg, step: int, rank: int, distributed: bool):
    """Save model checkpoint from rank 0; sync if distributed."""
    if distributed:
        dist.barrier()
    if rank == 0:
        try:
            output_dir = Path(cfg.output_dir).expanduser() / f"step-{step}"
            output_dir.mkdir(parents=True, exist_ok=True)
            model_to_save = model
            while hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module
            while hasattr(model_to_save, "_orig_mod"):
                model_to_save = model_to_save._orig_mod
            try:
                model_to_save.save_pretrained(str(output_dir))
            except Exception:
                model_to_save.save_pretrained(str(output_dir), safe_serialization=False)
            torch.save(
                {
                    "cfg": asdict(cfg),
                    "steps": step,
                },
                output_dir / "trainer_state.pt",
            )
            print(f"[ckpt] checkpoint saved at step {step} -> {output_dir}", flush=True)
        except Exception as e:
            print(f"[ckpt] failed to save at step {step} to {output_dir}: {e}", flush=True)
    if distributed:
        dist.barrier()


def main() -> None:
    cfg = Config()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA-capable devices.")
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        torch.cuda.set_device(local_rank)
        backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl")
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
            device_id=local_rank,
            timeout=timedelta(minutes=30),
        )
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)

    if rank == 0:
        print("[run] resolving model path...", flush=True)
    base_path = resolve_model_path(cfg.base_path)

    if rank == 0:
        print("[run] loading teacher model (base Olmo3)...", flush=True)
    teacher = load_model(base_path, device, dtype)
    freeze_model(teacher)

    if rank == 0:
        print("[run] loading student checkpoint...", flush=True)
    student = load_model("/workspace/a40/checkpoints/r2-408", device, dtype)
    student_base = student  # may be wrapped later; keep reference for bias checks
    layer_param_groups = prepare_layers(
        student,
        cfg.lr,
        cfg.log_act_s_lr_mult,
        cfg.log_weight_s_lr_mult,
        cfg.vocab_bias_lr_mult,
        cfg.down_proj_bias_lr_mult,
    )

    def _zero_nonfinite_biases(student_base, *, tag: str):
        cleaned: list[str] = []
        head_bias = getattr(student_base.lm_head, "bias", None)
        if head_bias is not None:
            before = head_bias.clone()
            head_bias.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if not torch.equal(before, head_bias):
                cleaned.append("lm_head.bias")
        for i, layer in enumerate(student_base.model.layers):
            b = getattr(layer.mlp, "down_proj").bias
            if b is None:
                continue
            before = b.clone()
            b.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if not torch.equal(before, b):
                cleaned.append(f"layer{i}.down_proj.bias")
        if cleaned and rank == 0:
            print(f"[bias] zeroed nonfinite biases ({tag}):", cleaned, flush=True)

    def _maybe_describe_bias_opt(student_base, optimizer):
        if os.environ.get("BIAS_CHECK", "0") != "1" or rank != 0:
            return
        head_bias = getattr(student_base.lm_head, "bias", None)
        target_params = {"lm_head.bias": head_bias}
        for i, layer in enumerate(student_base.model.layers):
            target_params[f"layer_{i}.down_proj.bias"] = getattr(layer.mlp, "down_proj").bias

        print("[bias] lr/weight_decay for bias params:", flush=True)
        for name, param in target_params.items():
            found = False
            for group in optimizer.param_groups:
                if any(param is p for p in group["params"]):
                    print(
                        f"  {name}: lr={group['lr']:.3e}, wd={group.get('weight_decay', 0.0)}",
                        flush=True,
                    )
                    found = True
                    break
            if not found:
                print(f"  {name}: NOT in optimizer", flush=True)

    def _maybe_log_bias_grads(student_base, step):
        if os.environ.get("BIAS_CHECK", "0") != "1" or rank != 0:
            return
        with torch.no_grad():
            head_bias = getattr(student_base.lm_head, "bias", None)
            head_grad = None if head_bias is None else head_bias.grad
            h_norm = None if head_grad is None else head_grad.norm().item()
            print(f"[bias_grad] step={step} lm_head.bias grad_norm={h_norm}", flush=True)
            # Log a few layers to avoid huge prints.
            for i, layer in enumerate(student_base.model.layers[:4]):
                b = getattr(layer.mlp, "down_proj").bias
                g = None if b is None else b.grad
                g_norm = None if g is None else g.norm().item()
                g_max = None if g is None else g.abs().max().item()
                b_norm = None if b is None else b.norm().item()
                req = None if b is None else b.requires_grad
                if b is not None and not torch.isfinite(b).all():
                    b.data.zero_()
                    print(f"[bias_grad] step={step} layer{i}.down_proj.bias had nonfinite values -> zeroed", flush=True)
                if g is not None and not torch.isfinite(g).all():
                    b.grad.zero_()
                    g_norm = 0.0
                    g_max = 0.0
                    print(f"[bias_grad] step={step} layer{i}.down_proj.bias had nonfinite grad -> zeroed", flush=True)
                print(
                    f"[bias_grad] step={step} layer{i}.down_proj.bias "
                    f"req_grad={req} param_norm={b_norm} grad_norm={g_norm} grad_max={g_max}",
                    flush=True,
                )

    def teacher_infer(input_ids, attention_mask):
        with torch.inference_mode():
            return teacher(input_ids=input_ids, attention_mask=attention_mask)

    if cfg.compile:
        if rank == 0:
            print("[run] compiling student with torch.compile...", flush=True)
        student = torch.compile(
            student,
            fullgraph=False,
            options={"triton.cudagraphs": False},
        )

    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        student = DDP(
            student,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    student_base = student.module if hasattr(student, "module") else student
    _zero_nonfinite_biases(student_base, tag="startup")
    optimizer = torch.optim.AdamW(layer_param_groups, weight_decay=cfg.weight_decay)
    _maybe_describe_bias_opt(student_base, optimizer)

    last_kl: torch.Tensor = torch.tensor(0.0, device=device, dtype=dtype)
    last_top1: float = 0.0
    last_flip: float = 0.0
    ema_kl: float = 0.0

    def set_layer_lrs(lr: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_mult", 1.0))

    progress = tqdm(
        total=None,
        desc="distill",
        dynamic_ncols=True,
        disable=rank != 0,
    )
    step = 0
    batch_iter = iter(
        build_dataloader(cfg, base_path, seed=cfg.seed, world_size=world_size, rank=rank)
    )

    while True:
        step += 1

        checkpoint_requested = maybe_update_lr_or_checkpoint(
            cfg,
            distributed=distributed,
            device=device,
            rank=rank,
        )
        if checkpoint_requested:
            save_checkpoint(student, cfg, step, rank, distributed)

        current_lr = cfg.lr
        set_layer_lrs(current_lr)

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(cfg.accumulate_steps):
            sync = micro_step == (cfg.accumulate_steps - 1)
            sync_ctx = nullcontext()
            if distributed and not sync:
                sync_ctx = student.no_sync()

            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(
                    build_dataloader(
                        cfg,
                        base_path,
                        seed=cfg.seed + step,
                        world_size=world_size,
                        rank=rank,
                    )
                )
                batch = next(batch_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with sync_ctx:
                student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
                teacher_logits = teacher_infer(input_ids=input_ids, attention_mask=attention_mask).logits

                kl_loss = compute_kl_loss(student_logits, teacher_logits, attention_mask)
                top1_acc, flip_pen = argmax_stability_metrics(teacher_logits, student_logits, attention_mask)
                loss = (cfg.kl_weight * kl_loss) / cfg.accumulate_steps
                loss.backward()

                last_kl = kl_loss.detach()
                last_top1 = top1_acc
                last_flip = flip_pen

                del student_logits, teacher_logits, loss

        _maybe_log_bias_grads(student_base, step)

        torch.nn.utils.clip_grad_norm_(student_base.parameters(), max_norm=1.0)
        optimizer.step()
        _zero_nonfinite_biases(student_base, tag="post_step")

        kl_val = float(last_kl.item())
        if distributed:
            kl_tensor = torch.tensor([kl_val], device=device)
            dist.all_reduce(kl_tensor, op=dist.ReduceOp.SUM)
            kl_val = kl_tensor.item() / world_size
        ema_kl = 0.6 * ema_kl + 0.4 * kl_val

        if rank == 0:
            progress.set_postfix(
                {
                    "kl": f"{kl_val:.4f}",
                    "ema_kl": f"{ema_kl:.4f}",
                    "base_lr": f"{current_lr:.2e}",
                    "step": f"{step}",
                    "top1": f"{last_top1:.3f}",
                    "flip": f"{last_flip:.3f}",
                }
            )

if __name__ == "__main__":
    main()
