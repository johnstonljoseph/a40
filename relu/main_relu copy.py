import argparse
import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from tqdm.auto import tqdm
from transformers.models.olmo3 import Olmo3ForCausalLM, Olmo3Model

from .data import build_dataloader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DIR = Path(__file__).resolve().parent


def load_mlp_calibration_values(path: Path) -> dict[int, tuple[float, float, float]]:
    values: dict[int, tuple[float, float, float]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Calibration values file not found: {path}")

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line == "":
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid calibration line: {raw_line!r}")
        layer_idx = int(parts[0])
        a_p = float(parts[1])
        a_m = float(parts[2])
        x0 = float(parts[3])
        values[layer_idx] = (a_p, a_m, x0)

    if len(values) == 0:
        raise ValueError(f"No calibration values parsed from: {path}")
    return values


@dataclass
class Config:
    steps: int = 2000
    batch_size: int = 5
    seq_len: int = 1024
    accumulate_steps: int = 8
    lr: float = 4e-5
    act_param_lr_mult: float = 1.0
    device: str = "cuda"
    dtype: str = "bfloat16"
    base_path: str = "/workspace/.hf_home/hub/models--allenai--Olmo-3-7B-Think"
    output_dir: str = str(DIR / "checkpoints" / "student_final")
    dataset_sft: Optional[str] = "allenai/Dolci-Think-SFT-7B"
    dataset_dpo: Optional[str] = "allenai/Dolci-Think-DPO-7B"
    dataset_rl: Optional[str] = "allenai/Dolci-Think-RL-7B"
    dataset_ratio_sft: float = 0.3
    dataset_ratio_dpo: float = 0.3
    dataset_ratio_rl: float = 0.4
    shuffle_buffer_size: int = 100
    seed: int = 42
    num_workers: int = 0
    blend_steps: Optional[int] = 20
    compile: bool = True


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument(
        "--act-param-lr-mult",
        type=float,
        default=Config.act_param_lr_mult,
        help="LR multiplier for BlendableActivation parameters (a_p/a_m/x0).",
    )
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
        act_param_lr_mult=args.act_param_lr_mult,
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
        k: float,
        delta: float,
        blend: torch.Tensor,
        gate: torch.Tensor,
        a_p: torch.Tensor,
        a_m: torch.Tensor,
        x0: torch.Tensor,
        y0: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(blend, gate, a_p, a_m, x0, y0)
        ctx.k = float(k)
        ctx.delta = float(delta)
        silu = F.silu(gate)

        piecewise = torch.where(
            gate >= x0,
            a_p * (gate - x0) - y0,
            a_m * (gate - x0) - y0,
        )
        return silu * (1.0 - blend) + piecewise * blend

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[None, None, None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (blend, gate, a_p, a_m, x0, y0) = ctx.saved_tensors
        delta = float(ctx.delta)
        blend = blend.to(dtype=gate.dtype)

        sig = torch.sigmoid(gate)
        dsilu = sig + gate * sig * (1.0 - sig)

        # Surrogate derivative: within [x0-delta, x0+delta] linearly interpolate slope.
        t = gate - x0
        d = torch.as_tensor(delta, dtype=gate.dtype, device=gate.device)

        left = (t <= -d).to(dtype=gate.dtype)
        right = (t >= d).to(dtype=gate.dtype)
        mid = 1.0 - left - right

        slope_mid = ((a_p - a_m) / (2.0 * d)) * t + (a_p + a_m) / 2.0
        slope = left * a_m + right * a_p + mid * slope_mid

        dact_dgate = dsilu * (1.0 - blend) + slope * blend
        grad_gate = grad_output * dact_dgate

        grad_piece = grad_output * blend

        # Parameter grads via a quadratic surrogate in the transition region (consistent with linear slope).
        d_a_p_mid = (t * t) * (1.0 / (4.0 * d)) + t * 0.5 + d * 0.25
        d_a_m_mid = (t * t) * (-1.0 / (4.0 * d)) + t * 0.5 - d * 0.25

        grad_a_p = (grad_piece * (right * t + mid * d_a_p_mid)).sum()
        grad_a_m = (grad_piece * (left * t + mid * d_a_m_mid)).sum()

        # For y = surrogate(t) - y0, d/dx0 = -d/dt (ignoring region boundary).
        grad_x0 = (grad_piece * (-slope)).sum()
        grad_y0 = (grad_piece * (-1.0)).sum()

        return None, None, None, grad_gate, grad_a_p, grad_a_m, grad_x0, grad_y0


class BlendableActivation(nn.Module):
    """Blend between SiLU (0.0) and ReLU (1.0)."""

    def __init__(
        self,
        k: float = 20.0,
        delta: float = 0.1,
        a_p: float = 1.0,
        a_m: float = -0.04,
        x0: float = -0.2,
        y0: float = 0.2,
    ) -> None:
        super().__init__()
        self.register_buffer("_blend", torch.tensor(0.0))
        self._a_p = nn.Parameter(torch.tensor(float(a_p)))
        self._a_m = nn.Parameter(torch.tensor(float(a_m)))
        self._x0 = nn.Parameter(torch.tensor(float(x0)))
        self._y0 = nn.Parameter(torch.tensor(float(y0)))
        self._k = float(k)
        self._delta = float(delta)

    def set_blend(self, value: float) -> None:
        self._blend.data.fill_(float(value))

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        gate = gate.to(self._blend.dtype)
        blend = self._blend.to(gate.dtype)
        a_p = self._a_p.to(dtype=gate.dtype)
        a_m = self._a_m.to(dtype=gate.dtype)
        x0 = self._x0.to(dtype=gate.dtype)
        y0 = self._y0.to(dtype=gate.dtype)
        return _BlendActivationFunction.apply(self._k, self._delta, blend, gate, a_p, a_m, x0, y0)


def patch_single_layer_activation(
    model: Olmo3Model,
    layer_index: int,
    calibration: dict[int, tuple[float, float, float]],
    k: float = 20.0,
) -> BlendableActivation:
    layers = model.layers
    if not (0 <= layer_index < len(layers)):
        raise ValueError(f"Invalid target layer {layer_index}; model has {len(layers)} layers.")
    mlp = getattr(layers[layer_index], "mlp", None)
    if mlp is None:
        raise ValueError(f"Layer {layer_index} is missing an MLP module.")

    if layer_index not in calibration:
        raise ValueError(f"Missing calibration values for layer {layer_index}")
    (a_p, a_m, x0) = calibration[layer_index]

    act_fn = BlendableActivation(k=k).to(
        device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype
    )
    mlp.act_fn = act_fn
    return act_fn


def patch_layer_activations(
    model: Olmo3Model,
    calibration: dict[int, tuple[float, float, float]],
    k: float = 20.0,
) -> list[BlendableActivation]:
    return [
        patch_single_layer_activation(model, idx, calibration=calibration, k=k)
        for idx in range(len(model.layers))
    ]


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

    print("[run] cloning student model...", flush=True)
    student = copy.deepcopy(teacher).to(device=device)
    freeze_model(student)

    calibration_path = DIR / "mlp_calibration" / "values_20.txt"
    calibration = load_mlp_calibration_values(calibration_path)

    act_fns = patch_layer_activations(student.model, calibration=calibration)
    trainable_params = target_mlp_parameters_multi(student.model, trainable=True)

    act_shape_params: list[nn.Parameter] = []
    for act_fn in act_fns:
        act_shape_params.extend([act_fn._a_p, act_fn._a_m, act_fn._x0, act_fn._y0])
    act_shape_param_ids = {id(p) for p in act_shape_params}
    mlp_params = [p for p in trainable_params if id(p) not in act_shape_param_ids]

    if cfg.compile:
        print("[run] compiling student with torch.compile...", flush=True)
        student = torch.compile(
            student,
            fullgraph=False,
            options={"triton.cudagraphs": False},
        )

    optimizer = torch.optim.AdamW(
        [
            {"params": mlp_params, "lr": cfg.lr, "weight_decay": 0.0},
            {
                "params": act_shape_params,
                "lr": cfg.lr * cfg.act_param_lr_mult,
                "weight_decay": 0.0,
            },
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
    )
    ensure_adamw_state_dtype(optimizer, dtype)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps, eta_min=cfg.lr * 0.1
    )

    dataloader = build_dataloader(cfg, base_path)
    batch_iter = iter(dataloader)

    progress = tqdm(
        range(cfg.steps),
        desc="blend (SiLU→ReLU)",
        dynamic_ncols=True,
    )

    last_kl: Optional[torch.Tensor] = None
    last_metrics: Optional[dict[str, torch.Tensor]] = None

    for step in progress:
        blend = activation_mix_at_step(step, cfg.blend_steps)
        for act_fn in act_fns:
            act_fn.set_blend(blend)

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

        progress.set_postfix(
            {
                "kl": f"{kl_val:.4f}",
                "top1": f"{top1_acc:.3f}",
                "flip_pen": f"{flip_pen:.3f}",
                "blend": f"{blend:.3f}",
            }
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = student
    while hasattr(model_to_save, "_orig_mod"):
        model_to_save = model_to_save._orig_mod
    model_to_save.save_pretrained(str(output_dir))
    torch.save(
        {
            "cfg": asdict(cfg),
            "steps": cfg.steps,
        },
        output_dir / "trainer_state.pt",
    )


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
