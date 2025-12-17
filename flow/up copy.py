import argparse
import copy
import os
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
from transformers.models.olmo3 import Olmo3ForCausalLM, Olmo3Model

from ..data import build_dataloader
from ..utils import (
    anneal_at_step,
    compute_kl_loss,
    freeze_model,
    load_model,
    resolve_model_path,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    batch_size: int = 8
    seq_len: int = 1024
    accumulate_steps: int = 4
    lr: float = 1e-3
    device: str = "cuda"
    dtype: str = "bfloat16"
    base_path: str = "/workspace/.hf_home/hub/models--allenai--Olmo-3-7B-Think"
    output_dir: str = str(DIR / "checkpoints" / "student_final")
    dataset_sft: Optional[str] = "allenai/Dolci-Think-SFT-7B"
    dataset_dpo: Optional[str] = "allenai/Dolci-Think-DPO-7B"
    dataset_rl: Optional[str] = "allenai/Dolci-Think-RL-7B"
    dataset_ratio_sft: float = 0.4
    dataset_ratio_dpo: float = 0.3
    dataset_ratio_rl: float = 0.3
    shuffle_buffer_size: int = 10000
    seed: int = 42
    num_workers: int = 0
    blend_steps: int = 200
    kl_threshold: float = 0.04
    compile: bool = True
    hidden_mse_weight: float = 1.0
    kl_weight: float = 2.0


class _BlendActivationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        blend: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(blend, gate)
        silu = F.silu(gate)
        identity = gate
        return silu * (1.0 - blend) + identity * blend

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[None, torch.Tensor]:
        (blend, gate) = ctx.saved_tensors
        blend = blend.to(dtype=gate.dtype)
        sig = torch.sigmoid(gate)
        dsilu = sig + gate * sig * (1.0 - sig)
        dact_dgate = dsilu * (1.0 - blend) + blend
        grad_gate = grad_output * dact_dgate
        return None, grad_gate


class BlendableActivation(nn.Module):
    """Blend between SiLU (0.0) and Identity (1.0)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_blend", torch.tensor(0.0))

    def set_blend(self, value: float) -> None:
        self._blend.data.fill_(float(value))

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        gate = gate.to(self._blend.dtype)
        blend = self._blend.to(gate.dtype)
        return _BlendActivationFunction.apply(blend, gate)


def _get_mlp(model: Olmo3Model, layer_index: int):
    layers = model.layers
    if not (0 <= layer_index < len(layers)):
        raise ValueError(f"Invalid target layer {layer_index}; model has {len(layers)} layers.")
    mlp = getattr(layers[layer_index], "mlp", None)
    if mlp is None:
        raise ValueError(f"Layer {layer_index} is missing an MLP module.")
    return mlp


def patch_layer_activation_and_params(
    model: Olmo3Model, layer_index: int
) -> tuple[BlendableActivation, list[nn.Parameter]]:
    mlp = _get_mlp(model, layer_index)
    act_fn = BlendableActivation().to(
        device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype
    )
    mlp.act_fn = act_fn

    params: list[nn.Parameter] = list(mlp.parameters())
    for p in params:
        p.requires_grad = True

    ln = getattr(model.layers[layer_index], "post_feedforward_layernorm")
    weight = getattr(ln, "weight")
    weight.requires_grad = True
    params.append(weight)

    return act_fn, params


def attach_single_post_ff_hook(
    model: Olmo3Model,
    layer_index: int,
    acts: dict[str, torch.Tensor],
    detach: bool,
):
    ln = getattr(model.layers[layer_index], "post_feedforward_layernorm")

    def hook(_module, _inp, output):
        acts["act"] = output.detach() if detach else output

    return ln.register_forward_hook(hook)


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
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)

    if rank == 0:
        print("[run] resolving model path...", flush=True)
    base_path = resolve_model_path(cfg.base_path)

    if rank == 0:
        print("[run] loading teacher model...", flush=True)
    teacher = load_model(base_path, device, dtype)
    freeze_model(teacher)

    if rank == 0:
        print("[run] cloning student model...", flush=True)
    student = copy.deepcopy(teacher).to(device=device)

    num_layers = len(student.model.layers)

    act_fns: list[BlendableActivation] = []
    layer_param_groups: list[dict] = []
    for idx in range(num_layers):
        act_fn, params = patch_layer_activation_and_params(student.model, idx)
        act_fns.append(act_fn)
        layer_param_groups.append({"params": params, "lr": cfg.lr, "weight_decay": 0.01})

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
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(layer_param_groups)

    last_mse: torch.Tensor = torch.tensor(0.0, device=device, dtype=dtype)
    last_kl: torch.Tensor = torch.tensor(0.0, device=device, dtype=dtype)
    last_top1: float = 0.0
    last_flip: float = 0.0
    ema_kl: float = 0.0
    ema_mse: float = 0.0

    def set_layer_lrs(lr: float, active_layer: int) -> None:
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        optimizer.param_groups[active_layer]["lr"] = lr

    def set_layer_requires_grad(active_layer: int) -> None:
        for idx, group in enumerate(layer_param_groups):
            requires_grad = idx == active_layer
            for p in group["params"]:
                p.requires_grad = requires_grad

    total_blend_units = num_layers * cfg.blend_steps
    progress = tqdm(
        total=total_blend_units,
        desc="blend (SiLU→Id)",
        dynamic_ncols=True,
        disable=rank != 0,
    )
    prev_units = 0
    step = 0
    batch_iter = iter(
        build_dataloader(cfg, base_path, seed=cfg.seed, world_size=world_size, rank=rank)
    )

    teacher_acts: dict[str, torch.Tensor] = {}
    student_acts: dict[str, torch.Tensor] = {}
    current_active_layer: int = -1

    def attach_all_post_ff_hooks(model: Olmo3Model, acts: dict[str, torch.Tensor], detach: bool):
        handles = []
        for idx in range(num_layers):
            ln = getattr(model.layers[idx], "post_feedforward_layernorm")

            def hook(_module, _inp, output, idx=idx):
                if idx == current_active_layer:
                    acts["act"] = output.detach() if detach else output

            handles.append(ln.register_forward_hook(hook))
        return handles

    teacher_hooks = attach_all_post_ff_hooks(teacher.model, teacher_acts, detach=True)
    student_base = student.module if hasattr(student, "module") else student
    student_hooks = attach_all_post_ff_hooks(student_base.model, student_acts, detach=False)

    try:
        for current_layer in reversed(range(num_layers)):
            current_active_layer = current_layer
            blend_progress = 0
            layer_start_step = step
            set_layer_requires_grad(current_layer)
            while blend_progress <= cfg.blend_steps:
                step += 1
                
                current_lr = cfg.lr * ema_kl
                set_layer_lrs(current_lr, current_layer)

                blend_val = anneal_at_step(blend_progress, cfg.blend_steps)
                act_fns[current_layer].set_blend(blend_val)

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
                        diff = student_acts["act"] - teacher_acts["act"]
                        mse_loss = (diff.pow(2) * mask).sum() / (mask.sum() * diff.shape[-1])

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
                ema_kl = 0.6 * ema_kl + 0.4 * kl_val
                ema_mse = 0.6 * ema_mse + 0.4 * mse_val

                advance = ema_kl <= cfg.kl_threshold
                if advance:
                    blend_progress += 1
                    current_units = (num_layers - 1 - current_layer) * cfg.blend_steps + min(blend_progress, cfg.blend_steps)
                    if rank == 0:
                        progress.update(current_units - prev_units)
                        prev_units = current_units
                if rank == 0:
                    progress.set_postfix(
                        {
                            "kl": f"{kl_val:.4f}",
                            "ema_kl": f"{ema_kl:.4f}",
                            "ema_mse": f"{ema_mse:.4f}",
                            "layer": f"{current_layer}",
                            "blend": f"{min(1.0, blend_progress / float(cfg.blend_steps)):.3f}",
                            "base_lr": f"{current_lr:.2e}",
                            "step": f"{step}",
                            "top1": f"{last_top1:.3f}",
                            "flip": f"{last_flip:.3f}",
                        }
                    )

            if rank == 0:
                layer_dir = Path(cfg.output_dir).parent
                layer_dir.mkdir(parents=True, exist_ok=True)
                model_to_save = student
                if hasattr(model_to_save, "module"):
                    model_to_save = model_to_save.module
                while hasattr(model_to_save, "_orig_mod"):
                    model_to_save = model_to_save._orig_mod
                layer_mod = model_to_save.model.layers[current_layer]
                state = {
                    "mlp.gate_proj.weight": layer_mod.mlp.gate_proj.weight.detach().cpu(),
                    "mlp.up_proj.weight": layer_mod.mlp.up_proj.weight.detach().cpu(),
                    "mlp.down_proj.weight": layer_mod.mlp.down_proj.weight.detach().cpu(),
                    "post_feedforward_layernorm.weight": layer_mod.post_feedforward_layernorm.weight.detach().cpu(),
                }
                layer_path = layer_dir / f"layer_{current_layer}.pt"
                torch.save(state, layer_path)
                layer_steps = step - layer_start_step
                print(f"[layer {current_layer}] finished in {layer_steps} steps", flush=True)
    finally:
        for h in teacher_hooks + student_hooks:
            h.remove()

    if distributed:
        dist.barrier()

    if rank == 0:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = student
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module
        while hasattr(model_to_save, "_orig_mod"):
            model_to_save = model_to_save._orig_mod
        model_to_save.save_pretrained(str(output_dir))
        torch.save(
            {
                "cfg": asdict(cfg),
                "steps": step,
            },
            output_dir / "trainer_state.pt",
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
