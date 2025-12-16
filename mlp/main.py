from .utils import load_model, freeze_model, resolve_model_path, ensure_adamw_state_dtype, maybe_compile, save_checkpoint
from a40.data import build_dataloader

@dataclass
class Config:
    steps: int = 2000
    batch_size: int = 5
    seq_len: int = 1024
    accumulate_steps: int = 8
    lr: float = 2e-5
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
    num_workers: int = 0
    blend_steps: Optional[int] = 20
    compile: bool = True


class Activation(nn.Module):
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


def target_mlp_parameters_multi(
    model: Olmo3Model, trainable: bool = True
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for idx in range(len(model.layers)):
        params.extend(target_mlp_parameters(model, idx, trainable=trainable))
    return params


def make_optimizer(cfg: Config, act_fns: list[Activation]) -> torch.optim.Optimizer:
    act_params: list[nn.Parameter] = []
    for act_fn in act_fns:
        act_params.extend([act_fn._a_p, act_fn._a_m, act_fn._x0, act_fn._y0])

    optimizer_groups = [
        {"params": other_params, "lr": cfg.lr, "weight_decay": 0.0},
        {"params": act_params, "lr": cfg.lr * cfg.act_lr_mult, "weight_decay": 0.0},
    ]
    beta2 = 0.95 if use_bf16 else 0.999
    eps = 1e-10 if use_bf16 else 1e-8
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(0.9, beta2),
        eps=eps,
        foreach=False,
    )
    ensure_adamw_state_dtype(optimizer, dtype)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps, eta_min=cfg.lr * 0.1
    )
    

def train(cfg: Config) -> None:
    progress = tqdm(
        range(start_step, cfg.steps),
        initial=start_step,
        total=cfg.steps,
        desc="distill",
        bar_format="{desc}: |{bar}| {n_fmt}/{total_fmt} {postfix}",
        dynamic_ncols=True,
        ncols=140,
        disable=(rank != 0),
    )

    batch_iter = iter(build_dataloader(cfg, base_path))
    for step in progess:
        blend = anneal_at_step(step, cfg.blend_steps)
        for act_fn in act_fns:
            act_fn.set_blend(blend)

        for micro in range(cfg.accumulate_steps):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(build_dataloader(cfg, base_path, step))
                batch = next(batch_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            kl_loss = compute_kl_loss(student_logits, teacher_logits, attention_mask)
            loss = kl_loss / cfg.accumulate_steps
            loss.backward()

            last_kl = kl_loss.detach().item()

            del student_logits, teacher_logits, loss

        optimizer.step()
        scheduler.step()

        progress.set_postfix(
            {
                "kl": f"{last_kl:.4f}",
                "blend": f"{blend:.3f}",
            }
        )

    save_checkpoint(cfg, student, optimizer, scheduler, step)













def main() -> None:
    cfg = Config()
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



    teacher, student = maybe_compile(teacher, student)
    

    optimizer = make_optimizer(cfg)
    

if __name__ == "__main__":
    main()
