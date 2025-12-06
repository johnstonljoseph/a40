import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass


@dataclass(frozen=True)
class ActivationCalibration:
    batch_size: int = 1
    seq_len: int = 1
    sample_count: int = 1
    percentile: float = 0.9999


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
        act_calib: ActivationCalibration | None = None,
    ):
        super().__init__()
        self.qmax = 1 << (bits - 1)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        self.log_act_s = nn.Parameter(torch.zeros(()))
        self.log_weight_s = nn.Parameter(torch.empty(out_features, 1))

        self.act_calib = act_calib
        if act_calib is not None:
            self.samples_collected = 0
            self.register_buffer(
                "act_samples",
                torch.empty(act_calib.sample_count, act_calib.batch_size, act_calib.seq_len, in_features),
                persistent=False,
            )
            self.set_trainable(False)

        

    def forward(self, x):
        if self.act_calib is not None and self.samples_collected < self.act_calib.sample_count:
            with torch.no_grad():
                sample = x.detach().to(dtype=self.act_samples.dtype)
                self.act_samples[self.samples_collected].copy_(sample)
                self.samples_collected += 1
                if self.samples_collected == self.act_calib.sample_count:
                    flat = self.act_samples.abs().flatten().to("cpu", dtype=torch.float32)
                    act_s = torch.quantile(flat, self.act_calib.percentile)
                    self.log_act_s.copy_(act_s.log())
                    self.set_trainable(True)
            return F.linear(x, self.weight)

        x_q = QuantFn.apply(x, self.log_act_s.exp(), self.qmax)
        w_q = QuantFn.apply(self.weight, self.log_weight_s.exp(), self.qmax)
        return F.linear(x_q, w_q)

    def set_weight_scales(self, scales: torch.Tensor) -> None:
        if scales.dim() != 1:
            raise ValueError()
        target = scales.view(-1, 1).to(dtype=self.log_weight_s.dtype, device=self.log_weight_s.device)
        with torch.no_grad():
            self.log_weight_s.copy_(target.log())

    def set_trainable(self, enabled: bool) -> None:
        self.weight.requires_grad = enabled
        self.log_weight_s.requires_grad = enabled
        self.log_act_s.requires_grad = enabled
            
