import torch
import torch.nn as nn
import pytest

import quant


def silq_loss(row_abs: torch.Tensor, b: float, s: torch.Tensor) -> torch.Tensor:
    """Autograd-friendly SiLQ objective."""
    inside = (s * s) / 12.0
    outside = torch.relu(row_abs - s * b).square()
    return torch.maximum(inside, outside).sum()

def optimize_s_with_grad(row_abs: torch.Tensor, b: float, steps: int = 400, lr: float = 5e-3):
    original_dtype = row_abs.dtype
    row_abs = row_abs.detach().to(torch.float64)
    s = (row_abs.mean() / max(b, 1e-8)).clamp(min=1e-5).detach().requires_grad_(True)
    opt = torch.optim.Adam([s], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = silq_loss(row_abs, b, s)
        loss.backward()
        opt.step()
        with torch.no_grad():
            s.clamp_(min=1e-8)
    return s.detach().to(original_dtype)

@pytest.mark.parametrize("num_weights", [4, 32, 128])
def test_solve_silq_scale_matches_backprop_optimum(num_weights, b=16-0.5):
    torch.manual_seed(0)
    row_abs = torch.randn(num_weights, dtype=torch.float32).abs()
    solved = quant.solve_silq_scale(row_abs, b)
    optimized = optimize_s_with_grad(row_abs, b)
    assert torch.allclose(solved, optimized, rtol=1e-2, atol=1e-4)

@pytest.mark.parametrize("in_features,out_features", [(3, 2), (64, 128)])
def test_quant_linear_initialize_matches_backprop_calibration(in_features, out_features):
    torch.manual_seed(0)
    linear = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.randn_like(linear.weight))

    batch_size = 1
    seq_len = 1
    quant_layer = quant.QuantLinear(in_features, out_features, batch_size, seq_len, bits=4)
    quant_layer.initialize(linear)

    b = float(quant_layer.qmax) - 0.5
    expected_scales = torch.stack([
        optimize_s_with_grad(row.abs(), b) for row in linear.weight
    ])

    calibrated_scales = quant_layer.log_weight_s.exp().squeeze(1)

    assert torch.allclose(calibrated_scales, expected_scales, rtol=1e-2, atol=1e-4)
