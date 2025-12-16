import argparse
import copy
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers.models.olmo3 import Olmo3ForCausalLM, Olmo3Model

from ..data import build_dataloader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    steps: int = 128000
    batch_size: int = 5
    seq_len: int = 1024
    accumulate_steps: int = 6
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
    shuffle_buffer_size: int = 100
    seed: int = 42
    num_workers: int = 0
    blend_steps: Optional[int] = 50
    kl_threshold: float = 0.05
    lr_gamma: float = 0.6
    compile: bool = True
    hidden_mse_weight: float = 1.0
    kl_weight: float = 1.0


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--dataset-sft", type=str, default=Config.dataset_sft)
    parser.add_argument("--dataset-dpo", type=str, default=Config.dataset_dpo)
    parser.add_argument("--dataset-rl", type=str, default=Config.dataset_rl)
    parser.add_argument("--dataset-ratio-sft", type=float, default=Config.dataset_ratio_sft)
    parser.add_argument("--dataset-ratio-dpo", type=float, default=Config.dataset_ratio_dpo)
    parser.add_argument("--dataset-ratio-rl", type=float, default=Config.dataset_ratio_rl)
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
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--dtype", type=str, default=Config.dtype)
    parser.add_argument(
        "--accumulate-steps",
        type=int,
        default=Config.accumulate_steps,
        help="Number of micro-batches to accumulate before an optimizer step.",
    )
    parser.add_argument(
        "--blend-steps",
        type=int,
        default=Config.blend_steps,
        help="Steps to anneal SiLU→ReLU blend.",
    )
    parser.add_argument(
        "--kl-threshold",
        type=float,
        default=Config.kl_threshold,
        help="EMA KL threshold to advance blend for the active layer.",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=Config.lr_gamma,
        help="Geometric LR decay factor per distance from active layer.",
    )
    parser.add_argument(
        "--mse-weight",
        type=float,
        default=Config.hidden_mse_weight,
        help="Weight for hidden-state MSE loss.",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=Config.kl_weight,
        help="Weight for final-token KL loss.",
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=Config.compile,
        help="Use torch.compile on the student model.",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile even if default is on.",
    )
    args = parser.parse_args()

    return Config(
        steps=args.steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        dataset_sft=args.dataset_sft,
        dataset_dpo=args.dataset_dpo,
        dataset_rl=args.dataset_rl,
        dataset_ratio_sft=args.dataset_ratio_sft,
        dataset_ratio_dpo=args.dataset_ratio_dpo,
        dataset_ratio_rl=args.dataset_ratio_rl,
        seq_len=args.seq_len,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
        num_workers=args.num_workers,
        dtype=args.dtype,
        accumulate_steps=args.accumulate_steps,
        blend_steps=args.blend_steps,
        kl_threshold=args.kl_threshold,
        lr_gamma=args.lr_gamma,
        hidden_mse_weight=args.mse_weight,
        kl_weight=args.kl_weight,
        compile=args.compile,
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


def activation_mix_at_step(step: int, end: Optional[int]) -> float:
    """Cosine anneal mixing coefficient in [0,1]: 0 -> start (SiLU), 1 -> end (ReLU)."""
    if end is None or end <= 0 or step >= end:
        return 1.0
    frac = step / float(end)
    return 0.5 * (1.0 - math.cos(math.pi * frac))


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


def patch_single_layer_activation(
    model: Olmo3Model,
    layer_index: int,
) -> BlendableActivation:
    layers = model.layers
    if not (0 <= layer_index < len(layers)):
        raise ValueError(f"Invalid target layer {layer_index}; model has {len(layers)} layers.")
    mlp = getattr(layers[layer_index], "mlp", None)
    if mlp is None:
        raise ValueError(f"Layer {layer_index} is missing an MLP module.")

    act_fn = BlendableActivation().to(
        device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype
    )
    mlp.act_fn = act_fn
    return act_fn


def patch_layer_activations(
    model: Olmo3Model,
    layer_indices: Sequence[int],
) -> list[BlendableActivation]:
    act_fns: list[BlendableActivation | None] = [None] * len(model.layers)
    for idx in layer_indices:
        act_fns[idx] = patch_single_layer_activation(model, idx)
    # Replace Nones with identity placeholders for unused layers to keep indexing simple
    for i, fn in enumerate(act_fns):
        if fn is None:
            # Install a non-trainable identity blend so indexing is total
            act_fns[i] = patch_single_layer_activation(model, i)
            act_fns[i].set_blend(1.0)
            for p in act_fns[i].parameters():
                p.requires_grad = False
    return act_fns  # indexed by layer id


def load_model(model_path: str, device: torch.device, dtype: torch.dtype) -> Olmo3ForCausalLM:
    model = Olmo3ForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.config.use_cache = False
    model.eval()
    model = model.to(device)
    return model


def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def target_mlp_parameters(model: Olmo3Model, layer_index: int, trainable: bool = True) -> list[nn.Parameter]:
    layers = model.layers
    if not (0 <= layer_index < len(layers)):
        raise ValueError(f"Invalid target layer {layer_index}; model has {len(layers)} layers.")
    mlp = getattr(layers[layer_index], "mlp", None)
    if mlp is None:
        raise ValueError(f"Layer {layer_index} is missing an MLP module.")
    params = list(mlp.parameters())
    for param in params:
        param.requires_grad = trainable
    return params


def target_mlp_parameters_multi(
    model: Olmo3Model, trainable: bool = True
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for idx in range(len(model.layers)):
        params.extend(target_mlp_parameters(model, idx, trainable=trainable))
    return params


def target_mlp_parameters_subset(
    model: Olmo3Model, layer_indices: Sequence[int], trainable: bool = True
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for idx in layer_indices:
        params.extend(target_mlp_parameters(model, idx, trainable=trainable))
    return params


def target_post_mlp_layernorm_scalars_subset(
    model: Olmo3Model, layer_indices: Sequence[int], trainable: bool = True
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for idx in layer_indices:
        ln = getattr(model.layers[idx], "post_feedforward_layernorm")
        weight = getattr(ln, "weight")
        weight.requires_grad = trainable
        params.append(weight)
    return params


def set_layer_trainable(
    model: Olmo3Model, layer_indices: Sequence[int], trainable: bool
) -> None:
    target_mlp_parameters_subset(model, layer_indices, trainable=trainable)
    target_post_mlp_layernorm_scalars_subset(model, layer_indices, trainable=trainable)


def ensure_adamw_state_dtype(optimizer: torch.optim.AdamW, target_dtype: torch.dtype) -> None:
    """Ensure AdamW state tensors live in the desired dtype (bf16 support)."""
    if target_dtype != torch.bfloat16:
        return

    for group in optimizer.param_groups:
        for param in group["params"]:
            if param is None or not param.requires_grad:
                continue
            if param.dtype != target_dtype:
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
    freeze_model(student)

    num_layers = len(student.model.layers)
    layer_indices = tuple(range(num_layers))

    act_fns = patch_layer_activations(student.model, layer_indices)

    # Build per-layer param groups (MLP weights + post-FF LN weight)
    layer_param_groups: list[dict] = []
    for idx in layer_indices:
        params = []
        params.extend(target_mlp_parameters_subset(student.model, [idx], trainable=True))
        params.extend(target_post_mlp_layernorm_scalars_subset(student.model, [idx], trainable=True))
        layer_param_groups.append(
            {"params": params, "lr": cfg.lr, "weight_decay": 0.0}
        )

    if cfg.compile:
        if rank == 0:
            print("[run] compiling teacher with torch.compile...", flush=True)
        teacher = torch.compile(
            teacher,
            fullgraph=False,
            options={"triton.cudagraphs": False},
        )
        if rank == 0:
            print("[run] compiling student with torch.compile...", flush=True)
        student = torch.compile(
            student,
            fullgraph=False,
            options={"triton.cudagraphs": False},
        )

    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        student = DDP(student, device_ids=[local_rank], output_device=local_rank)

    # optimizer = torch.optim.AdamW(
    #     layer_param_groups,
    #     betas=(0.9, 0.95),
    #     eps=1e-8,
    #     foreach=False,
    # )
    # ensure_adamw_state_dtype(optimizer, dtype)
    optimizer = torch.optim.AdamW(
        layer_param_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # # Hook helpers to capture post-FFN norm activations
    # def attach_post_ff_hooks(model: Olmo3Model, detach: bool) -> tuple[dict[int, torch.Tensor], list]:
    #     acts: dict[int, torch.Tensor] = {}
    #     handles: list = []
    #     for idx in range(len(model.layers)):
    #         ln = getattr(model.layers[idx], "post_feedforward_layernorm")

    #         def hook(_module, _inp, output, idx=idx):
    #             acts[idx] = output.detach() if detach else output

    #         handles.append(ln.register_forward_hook(hook))
    #     return acts, handles

    # teacher_acts, teacher_hooks = attach_post_ff_hooks(teacher.model, detach=True)
    # student_acts, student_hooks = attach_post_ff_hooks(student.model, detach=False)

    teacher_hooks: list = []
    student_hooks: list = []

    last_kl: Optional[torch.Tensor] = None
    last_metrics: Optional[dict[str, torch.Tensor]] = None
    ema_kl: float = 0.0
    # ema_mse: float = 0.0

    def set_layer_lrs(base_lr: float, active_layer: int) -> None:
        for i, group in enumerate(optimizer.param_groups):
            group["lr"] = base_lr if i == active_layer else 0.0

    def set_layer_blend(active_layer: int, blend_step: int) -> None:
        blend_val = activation_mix_at_step(blend_step, cfg.blend_steps)
        act_fns[active_layer].set_blend(blend_val)

    num_layers = len(student.model.layers)
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

    for current_layer in reversed(range(num_layers)):
        blend_progress = 0
        while blend_progress < cfg.blend_steps:
            step += 1
            blend_val = activation_mix_at_step(blend_progress, cfg.blend_steps)
            base_lr = cfg.lr * ema_kl

            set_layer_lrs(base_lr, current_layer)
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
                    with torch.no_grad():
                        teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

                    # MSE temporarily disabled
                    mse_loss = torch.tensor(0.0, device=device, dtype=student_logits.dtype)

                    kl_loss = compute_kl_loss(student_logits, teacher_logits, attention_mask)
                    loss = (cfg.hidden_mse_weight * mse_loss + cfg.kl_weight * kl_loss) / cfg.accumulate_steps
                    loss.backward()

                    with torch.no_grad():
                        last_metrics = argmax_stability_metrics(teacher_logits, student_logits, attention_mask)
                    last_kl = kl_loss.detach()

                    del student_logits, teacher_logits, loss

            optimizer.step()

            kl_val = float(last_kl.item()) if last_kl is not None else 0.0
            ema_kl = 0.8 * ema_kl + 0.2 * kl_val
            # mse_val = float(mse_loss.item()) if mse_loss is not None else 0.0
            # ema_mse = mse_val if ema_mse is None else 0.9 * ema_mse + 0.1 * mse_val
            top1_acc = float(last_metrics["top1_acc"]) if last_metrics is not None else 0.0
            flip_pen = float(last_metrics["avg_flip_penalty"]) if last_metrics is not None else 0.0

            blend_progress += 1
            current_units = (num_layers - 1 - current_layer) * cfg.blend_steps + min(blend_progress, cfg.blend_steps)
            if current_units > total_blend_units:
                current_units = total_blend_units
            if rank == 0:
                progress.update(current_units - prev_units)
                prev_units = current_units
                progress.set_postfix(
                    {
                        "kl": f"{kl_val:.4f}",
                        "ema_kl": f"{ema_kl:.4f}",
                        # "ema_mse": f\"{ema_mse:.4f}\",
                        "top1": f"{top1_acc:.3f}",
                        "flip_pen": f"{flip_pen:.3f}",
                        "layer": f"{current_layer}",
                        "blend": f"{min(1.0, blend_progress / float(cfg.blend_steps)):.3f}",
                        "base_lr": f"{base_lr:.2e}",
                    }
                )

        if rank == 0:
            output_dir = Path(cfg.output_dir) / "layer_weights"
            output_dir.mkdir(parents=True, exist_ok=True)
            model_to_save = student
            if hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module
            while hasattr(model_to_save, "_orig_mod"):
                model_to_save = model_to_save._orig_mod
            layer_mod = model_to_save.model.layers[current_layer]
            state = {
                "mlp.gate_proj.weight": layer_mod.mlp.gate_proj.weight.detach().cpu(),
                "mlp.down_proj.weight": layer_mod.mlp.down_proj.weight.detach().cpu(),
                "mlp.up_proj.weight": layer_mod.mlp.up_proj.weight.detach().cpu(),
                "post_feedforward_layernorm.weight": layer_mod.post_feedforward_layernorm.weight.detach().cpu(),
            }
            torch.save(state, output_dir / f"layer_{current_layer}.pt")

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

    # Clean up hooks
    for h in teacher_hooks + student_hooks:
        h.remove()

    if distributed:
        dist.destroy_process_group()


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
