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
import torch.nn.functional as F
from torch.autograd import Function
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
from .activation import PiecewiseActivation, IdentityActivation

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DIR = Path(__file__).resolve().parent

@dataclass
class Config:
    batch_size: int = 6
    seq_len: int = 1024
    accumulate_steps: int = 8
    lr: float = 8e-5
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
    shuffle_buffer_size: int = 10
    seed: int = 43
    num_workers: int = 1
    compile: bool = True
    kl_threshold: float = 0.1
    hidden_mse_weight: float = 2.0
    kl_weight: float = 1.0
    weight_decay: float = 0.05
    log_act_s_lr_mult: float = 20.0
    log_weight_s_lr_mult: float = 1.0


def weight_calib_path(layer_index: int, name: str):
    weight_scales_dir = DIR.parent / "weight_calibration" / "values"
    return weight_scales_dir / f"{layer_index}" / f"{name}.pt"


def prepare_layers(
    model: MyOlmo3ForCausalLM,
    lr: float,
    log_act_s_lr_mult: float,
    log_weight_s_lr_mult: float,
) -> list[dict]:
    layer_param_groups: list[dict] = []
    for layer_index, layer in enumerate(model.model.layers):
        norm = getattr(layer, "post_feedforward_layernorm")
        norm_params = [getattr(norm, "weight")]
        layer_param_groups.append({"params": norm_params, "lr": lr, "lr_mult": 1.0})

        mlp = layer.mlp
        names = ["gate_proj", "up_proj", "down_proj"]
        for name in names:
            linear = getattr(mlp, name)
            linear.set_trainable(True)

            layer_param_groups.append({"params": [linear.weight], "lr": lr, "lr_mult": 1.0})
            layer_param_groups.append({"params": [linear.log_act_s], "lr": lr * log_act_s_lr_mult, "lr_mult": log_act_s_lr_mult})
            layer_param_groups.append({"params": [linear.log_weight_s], "lr": lr * log_weight_s_lr_mult, "lr_mult": log_weight_s_lr_mult})
            linear.q_name = f"layer{layer_index}.{mlp.__class__.__name__}.{name}"

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
    layer_param_groups = prepare_layers(
        student,
        cfg.lr,
        cfg.log_act_s_lr_mult,
        cfg.log_weight_s_lr_mult,
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
    optimizer = torch.optim.AdamW(layer_param_groups, weight_decay=cfg.weight_decay)

    last_mse: torch.Tensor = torch.tensor(0.0, device=device, dtype=dtype)
    last_kl: torch.Tensor = torch.tensor(0.0, device=device, dtype=dtype)
    last_top1: float = 0.0
    last_flip: float = 0.0
    ema_kl: float = 0.0
    ema_mse: float = 0.0

    def set_layer_lrs(lr: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_mult", 1.0))

    num_layers = len(student_base.model.layers)
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

    teacher_acts: list[torch.Tensor | None] = [None] * num_layers
    student_acts: list[torch.Tensor | None] = [None] * num_layers

    def attach_all_post_ff_hooks(model: Olmo3Model, acts: dict[str, torch.Tensor], detach: bool):
        handles = []
        for idx in range(num_layers):
            ln = getattr(model.layers[idx], "post_feedforward_layernorm")

            def hook(_module, _inp, output, idx=idx):
                acts[idx] = output.detach() if detach else output

            handles.append(ln.register_forward_hook(hook))
        return handles

    teacher_hooks = attach_all_post_ff_hooks(teacher.model, teacher_acts, detach=True)
    student_base = student.module if hasattr(student, "module") else student
    student_hooks = attach_all_post_ff_hooks(student_base.model, student_acts, detach=False)

    try:
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

                    mask = attention_mask.to(student_logits.dtype).unsqueeze(-1)
                    mse_accum = 0.0
                    for idx in range(num_layers):
                        s_act = student_acts[idx]
                        t_act = teacher_acts[idx]
                        if s_act is None or t_act is None:
                            raise RuntimeError(f"Missing activations for layer {idx}")
                        diff = s_act - t_act
                        mse_accum = mse_accum + (diff.pow(2) * mask).sum() / (mask.sum() * diff.shape[-1])
                    mse_loss = mse_accum / num_layers

                    kl_loss = compute_kl_loss(student_logits, teacher_logits, attention_mask)
                    top1_acc, flip_pen = argmax_stability_metrics(teacher_logits, student_logits, attention_mask)
                    loss = (cfg.hidden_mse_weight * mse_loss + cfg.kl_weight * kl_loss) / cfg.accumulate_steps
                    loss.backward()

                    last_mse = mse_loss.detach()
                    last_kl = kl_loss.detach()
                    last_top1 = top1_acc
                    last_flip = flip_pen

                    del student_logits, teacher_logits, loss

            optimizer.step()

            kl_val = float(last_kl.item())
            mse_val = float(last_mse.item())
            if distributed:
                kl_tensor = torch.tensor([kl_val], device=device)
                dist.all_reduce(kl_tensor, op=dist.ReduceOp.SUM)
                kl_val = kl_tensor.item() / world_size
            ema_kl = 0.6 * ema_kl + 0.4 * kl_val
            ema_mse = 0.6 * ema_mse + 0.4 * mse_val

            if rank == 0:
                progress.set_postfix(
                    {
                        "kl": f"{kl_val:.4f}",
                        "ema_kl": f"{ema_kl:.4f}",
                        "ema_mse": f"{ema_mse:.4f}",
                        "base_lr": f"{current_lr:.2e}",
                        "step": f"{step}",
                        "top1": f"{last_top1:.3f}",
                        "flip": f"{last_flip:.3f}",
                    }
                )
    finally:
        for h in teacher_hooks + student_hooks:
            h.remove()

if __name__ == "__main__":
    main()
