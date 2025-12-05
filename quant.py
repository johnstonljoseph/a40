import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm


def sum_like(tensor, s_shape) -> torch.Tensor:
    # for activations
    if len(s_shape) == 0:
        return tensor.sum()
    # for weights
    if len(s_shape) == 2:
        if tensor.dim() != 2:
            raise ValueError()
        if tensor.shape[0] != s_shape[0]:
            raise ValueError()
        return tensor.sum(1, keepdim=True)


def scale_grad(grad_out, y, q, s, qmax):
    coef = torch.where(
        y < -qmax,
        torch.tensor(-qmax, dtype=y.dtype, device=y.device),
        torch.where(
            qmax-1 < y,
            torch.tensor(qmax-1, dtype=y.dtype, device=y.device),
            q - y,
        )
    )
    grad_s_raw = sum_like(coef * grad_out, s.shape)
    elems_per_scale = y.numel() / s.numel()
    grad_scale = 1.0 / math.sqrt(elems_per_scale * qmax)
    return grad_s_raw * grad_scale


class QuantFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, qmax) -> torch.Tensor:
        # x: [batch_size, seq_len, hidden_size]
        # s: [] if per tensor, [out_features] if per channel
        y = x / s
        q = y.clamp(-qmax, qmax-1).round()
        xhat = q * s

        ctx.save_for_backward(x, q, s)
        ctx.qmax = qmax
        return xhat

    @staticmethod
    def backward(ctx, grad_out) -> torch.Tensor:
        x, q, s = ctx.saved_tensors
        qmax = ctx.qmax
        y = x / s

        mask = (-qmax <= y) & (y <= qmax-1)
        grad_x = grad_out * mask.to(grad_out.dtype)
        grad_s = scale_grad(grad_out, y, q, s, qmax)
        return grad_x, grad_s, None


class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bits: int = 8,
        sample_count: int = 5,
        percentile: float = 0.9999,
    ):
        super().__init__()
        self.qmax = 1 << (bits - 1)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        self.log_act_s = nn.Parameter(torch.zeros(()))
        self.log_weight_s = nn.Parameter(torch.empty(out_features, 1))

        self.percentile = percentile
        self.sample_count = sample_count
        self.samples_collected = 0
        self.register_buffer("act_samples", torch.empty(sample_count, in_features))

    def forward(self, x):
        if self.samples_collected < self.sample_count:
            with torch.no_grad():
                self.act_samples[self.samples_collected] = x.detach()
                self.samples_collected += 1
                if self.samples_collected == self.sample_count:
                    act_s = torch.quantile(self.act_samples.abs().flatten(), self.percentile)
                    self.log_act_s.copy_(act_s.log())
            return F.linear(x, self.weight)

        x_q = QuantFn.apply(x, self.log_act_s.exp(), self.qmax)
        w_q = QuantFn.apply(self.weight, self.log_weight_s.exp(), self.qmax)
        return F.linear(x_q, w_q)

    def initialize(self, linear: nn.Linear, *, show_progress: bool = False, desc: str | None = None) -> None:
        with torch.no_grad():
            if self.weight.data_ptr() != linear.weight.data_ptr():
                self.weight.copy_(linear.weight)

            b = float(self.qmax) - 0.5
            rows = self.weight
            progress = None
            if show_progress:
                progress = tqdm(
                    total=rows.shape[0],
                    desc=desc or "calibrating scales",
                    leave=False,
                )

            scales = []
            for row in rows:
                scales.append(solve_silq_scale(row.abs(), b))
                if progress is not None:
                    progress.update(1)

            if progress is not None:
                progress.close()

            self.log_weight_s.copy_(torch.stack(scales).unsqueeze(1).log())
            

def solve_silq_scale(row_abs: torch.Tensor, b: float) -> torch.Tensor:
    """
    Return the SiLQ step size s that minimizes

        F(s) = sum_i max(s^2/12, H(|w_i| - s*b) * (|w_i| - s*b)^2)

    for a single channel (1D tensor of magnitudes).
    """
    orig_dtype = row_abs.dtype
    device = row_abs.device

    # Work in float64
    work = row_abs.abs().to(torch.float64)
    n = work.numel()

    # Sort |w| descending and compute prefix sums:
    # sorted_vals[0] is largest |w|, prefix[k] = sum of top k magnitudes
    sorted_vals, _ = torch.sort(work, descending=True)
    prefix = torch.cat([sorted_vals.new_zeros(1), sorted_vals.cumsum(0)])  # len n+1

    # k = number of “clipped” / outer-branch weights (0..n)
    idx_long = torch.arange(n + 1, device=device, dtype=torch.long)
    k = idx_long.to(torch.float64)

    # For a fixed k, assuming top-k are in the outer quadratic and others inside,
    # derivative root is:
    #   s_k = (2 b * sum_{i=1..k} |w|_(i)) / ((n - k)/6 + 2 b^2 k)
    denom = (n - k) / 6.0 + 2.0 * (b ** 2) * k
    valid = denom > 0
    num = 2.0 * b * prefix[idx_long]
    safe_div = torch.where(valid, num / denom, torch.zeros_like(denom))
    s_from_k = safe_div
    # Keep only positive stationary points (avoid boolean indexing for vmap)
    s_stationary = torch.where(s_from_k > 0, s_from_k, torch.zeros_like(s_from_k))

    # Hinge breakpoints: where (|w| - s*b)^2 = s^2/12 and diff > 0
    # => |w| = s * (b + 1/sqrt(12)) => s = |w| / (b + 1/sqrt(12))
    inv_sqrt12 = 1.0 / math.sqrt(12.0)
    clip_den = b + inv_sqrt12
    if clip_den > 0:
        thresholds = sorted_vals / clip_den
        # Only care about positive thresholds
        thresholds = thresholds[thresholds > 0]
    else:
        thresholds = work.new_empty(0)

    # Always include a tiny positive epsilon near 0 as boundary candidate
    eps = work.new_tensor([1e-12])

    # Candidate pool: stationary points + hinge breakpoints + epsilon
    if s_stationary.numel() == 0 and thresholds.numel() == 0:
        candidate_pool = eps
    else:
        candidate_pool = torch.cat([s_stationary, thresholds, eps])

    # Evaluate the *true* loss for all candidates (vectorized)
    def _loss_batch(s_vals: torch.Tensor) -> torch.Tensor:
        s = s_vals.view(-1, 1)  # [M, 1]
        diff = work - s * b     # [M, n]
        inside = (s * s) / 12.0 # [M, 1]
        outside = torch.relu(diff).square()
        return torch.maximum(inside, outside).sum(dim=1)  # [M]

    candidate_pool = candidate_pool.clamp_min(1e-12)
    losses = _loss_batch(candidate_pool)
    best_idx = torch.argmin(losses)
    s_best = candidate_pool[best_idx]

    return s_best.clamp_min(1e-12).to(orig_dtype)