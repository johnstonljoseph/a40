from __future__ import annotations

import math
import torch


def solve(row_abs: torch.Tensor, b: float) -> torch.Tensor:
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