from __future__ import annotations

import math

import torch


def compute_weight_scales(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-row max divided by (qmax - 0.5), matching QuantLinear.initialize."""
    qmax = 1 << (bits - 1)
    b = float(qmax) - 0.5
    return weight.abs().amax(dim=1).clamp_min(1e-12) / b


def solve_silq_scale(row_abs: torch.Tensor, b: float) -> torch.Tensor:
    """
    Return the SiLQ step size s that minimizes

        F(s) = sum_i max(s^2/12, H(|w_i| - s*b) * (|w_i| - s*b)^2)

    for a single channel (1D tensor of magnitudes).
    """
    orig_dtype = row_abs.dtype
    device = row_abs.device

    work = row_abs.abs().to(torch.float64)
    n = work.numel()

    sorted_vals, _ = torch.sort(work, descending=True)
    prefix = torch.cat([sorted_vals.new_zeros(1), sorted_vals.cumsum(0)])

    idx_long = torch.arange(n + 1, device=device, dtype=torch.long)
    k = idx_long.to(torch.float64)

    denom = (n - k) / 6.0 + 2.0 * (b ** 2) * k
    valid = denom > 0
    num = 2.0 * b * prefix[idx_long]
    safe_div = torch.where(valid, num / denom, torch.zeros_like(denom))
    s_stationary = torch.where(safe_div > 0, safe_div, torch.zeros_like(safe_div))

    inv_sqrt12 = 1.0 / math.sqrt(12.0)
    clip_den = b + inv_sqrt12
    if clip_den > 0:
        thresholds = sorted_vals / clip_den
        thresholds = thresholds[thresholds > 0]
    else:
        thresholds = work.new_empty(0)

    eps = work.new_tensor([1e-12])
    if s_stationary.numel() == 0 and thresholds.numel() == 0:
        candidate_pool = eps
    else:
        candidate_pool = torch.cat([s_stationary, thresholds, eps])

    candidate_pool = candidate_pool.clamp_min(1e-12)

    def loss_batch(scales: torch.Tensor) -> torch.Tensor:
        s = scales.view(-1, 1)
        diff = work - s * b
        inside = (s * s) / 12.0
        outside = torch.relu(diff).square()
        return torch.maximum(inside, outside).sum(dim=1)

    losses = loss_batch(candidate_pool)
    best_idx = torch.argmin(losses)
    return candidate_pool[best_idx].clamp_min(1e-12).to(orig_dtype)
