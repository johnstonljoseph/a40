import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers.models.olmo3 import Olmo3ForCausalLM, Olmo3Model

from .data import build_dataloader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DIR = Path(__file__).resolve().parent
@dataclass
class Config:
    steps: int = 1000
    batch_size: int = 6
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
    dataset_ratio_sft: float = 0.4
    dataset_ratio_dpo: float = 0.3
    dataset_ratio_rl: float = 0.3
    shuffle_buffer_size: int = 10000
    seed: int = 42
    num_workers: int = 0
    compile: bool = True
    student_path: Optional[str] = "/workspace/a40/checkpoints/student_final_v1"


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
        "--student-path",
        type=str,
        default=Config.student_path,
        help="Optional model path to initialize the student from instead of cloning the teacher.",
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
        compile=args.compile,
        student_path=args.student_path,
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


def replace_mlp_activation_with_identity(model: Olmo3Model) -> None:
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        mlp.act_fn = nn.Identity()


def load_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Olmo3ForCausalLM:
    model = Olmo3ForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.config.use_cache = False
    model.eval()
    model = model.to(device)
    return model


def load_state_dict_from_disk(model_path: str) -> dict[str, torch.Tensor]:
    """Load a state dict (CPU) from a save_pretrained directory (supports sharded indexes)."""
    path = Path(model_path)
    bin_path = path / "pytorch_model.bin"
    bin_index = path / "pytorch_model.bin.index.json"
    safe_path = path / "model.safetensors"
    safe_index = path / "model.safetensors.index.json"

    def load_sharded(index_path: Path, load_fn):
        with index_path.open("r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        state: dict[str, torch.Tensor] = {}
        for param_name, shard_file in weight_map.items():
            shard_state = load_fn(path / shard_file)
            state[param_name] = shard_state[param_name]
        return state

    if bin_index.exists():
        return load_sharded(bin_index, lambda p: torch.load(p, map_location="cpu"))
    if safe_index.exists():
        try:
            from safetensors.torch import load_file as safetensors_load_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("safetensors is required to load the provided student checkpoint.") from exc
        return load_sharded(safe_index, lambda p: safetensors_load_file(str(p), device="cpu"))

    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")

    if safe_path.exists():
        try:
            from safetensors.torch import load_file as safetensors_load_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("safetensors is required to load the provided student checkpoint.") from exc
        return safetensors_load_file(str(safe_path), device="cpu")

    raise FileNotFoundError(
        f"Expected either {bin_path.name}, {safe_path.name}, or an index json under {model_path}"
    )


def strip_activation_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove leftover activation params (_a_p/_a_m/_x0/_y0/_blend) from older checkpoints."""
    return {k: v for k, v in state_dict.items() if ".mlp.act_fn." not in k}


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


def target_post_mlp_layernorm_scalars_multi(
    model: Olmo3Model, trainable: bool = True
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for layer in model.layers:
        ln = getattr(layer, "post_feedforward_layernorm")
        weight = getattr(ln, "weight")
        weight.requires_grad = trainable
        params.append(weight)
    return params


def target_mlp_parameters_multi(
    model: Olmo3Model, trainable: bool = True
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for idx in range(len(model.layers)):
        params.extend(target_mlp_parameters(model, idx, trainable=trainable))
    return params


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
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)

    print("[run] resolving model path...", flush=True)
    base_path = resolve_model_path(cfg.base_path)

    print("[run] loading teacher model...", flush=True)
    teacher = load_model(base_path, device, dtype)
    freeze_model(teacher)

    if cfg.student_path is None:
        print("[run] cloning student model...", flush=True)
        student = copy.deepcopy(teacher).to(device=device)
    else:
        print("[run] loading student model...", flush=True)
        student_path = resolve_model_path(cfg.student_path)
        full_state = load_state_dict_from_disk(student_path)
        base_state = strip_activation_keys(full_state)
        student = load_model(base_path, device, dtype)
        missing, unexpected = student.load_state_dict(base_state, strict=False)
        if unexpected:
            print(f"[run] warning: unexpected keys when loading student: {unexpected}")
        if missing:
            print(f"[run] warning: missing keys when loading student: {missing}")
    freeze_model(student)

    replace_mlp_activation_with_identity(student.model)

    trainable_params = target_mlp_parameters_multi(student.model, trainable=True)

    ln_params = target_post_mlp_layernorm_scalars_multi(student.model, trainable=True)

    mlp_params = trainable_params

    if cfg.compile:
        print("[run] compiling student with torch.compile...", flush=True)
        student = torch.compile(
            student,
            fullgraph=False,
            options={"triton.cudagraphs": False},
        )

    optimizer = torch.optim.AdamW(
        [
            {"params": mlp_params, "lr": cfg.lr, "weight_decay": 0.01},
            {"params": ln_params, "lr": cfg.lr, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
    )
    ensure_adamw_state_dtype(optimizer, dtype)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps, eta_min=cfg.lr * 0.01
    )

    dataloader = build_dataloader(cfg, base_path)
    batch_iter = iter(dataloader)

    progress = tqdm(range(cfg.steps), desc="distill", dynamic_ncols=True)

    last_kl: Optional[torch.Tensor] = None
    last_metrics: Optional[dict[str, torch.Tensor]] = None
    ema_kl: Optional[float] = None

    for step in progress:
        optimizer.zero_grad(set_to_none=True)

        for _ in range(cfg.accumulate_steps):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                batch = next(batch_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            kl_loss = compute_kl_loss(student_logits, teacher_logits, attention_mask)
            loss = kl_loss / cfg.accumulate_steps
            loss.backward()

            with torch.no_grad():
                last_metrics = argmax_stability_metrics(teacher_logits, student_logits, attention_mask)
            last_kl = kl_loss.detach()

            del student_logits, teacher_logits, loss

        optimizer.step()
        scheduler.step()

        kl_val = float(last_kl.item()) if last_kl is not None else 0.0
        top1_acc = float(last_metrics["top1_acc"]) if last_metrics is not None else 0.0
        flip_pen = float(last_metrics["avg_flip_penalty"]) if last_metrics is not None else 0.0
        ema_kl = kl_val if ema_kl is None else 0.9 * ema_kl + 0.1 * kl_val
        ema_kl_val = ema_kl if ema_kl is not None else kl_val

        progress.set_postfix(
            {
                "kl": f"{kl_val:.4f}",
                "ema_kl": f"{ema_kl_val:.4f}",
                "top1": f"{top1_acc:.3f}",
                "flip_pen": f"{flip_pen:.3f}",
            }
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = student
    while hasattr(model_to_save, "_orig_mod"):
        model_to_save = model_to_save._orig_mod
    model_to_save.save_pretrained(str(output_dir))


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
