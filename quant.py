import torch
import math
import torch.nn.functional as F
import torch.nn as nn


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
        # x_h = hadamard_transform(x.contiguous())
        # y = x / s
        # q = y.clamp(-qmax, qmax-1).round()
        # xhat_h = q * s
        # xhat = hadamard_transform(xhat_h.contiguous(), transpose=True)
        y = x / s
        q = y.clamp(-qmax, qmax-1).round()
        xhat = q * s

        ctx.save_for_backward(y, q, s)
        ctx.qmax = qmax
        return xhat

    @staticmethod
    def backward(ctx, grad_out) -> torch.Tensor:
        y, q, s = ctx.saved_tensors
        qmax = ctx.qmax

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
    ):
        super().__init__()
        self.qmax = 1 << (bits - 1)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        self.log_act_s = nn.Parameter(torch.empty(()))
        self.log_weight_s = nn.Parameter(torch.empty(out_features, 1))
        self.last_max_act: torch.Tensor | None = None

    def forward(self, x):
        act_scale = self.log_act_s.exp().to(x.dtype)
        weight_scale = self.log_weight_s.exp().to(self.weight.dtype)
        x_q = QuantFn.apply(x, act_scale, self.qmax)
        w_q = QuantFn.apply(self.weight, weight_scale, self.qmax)
        y = F.linear(x_q, w_q)
        # y = F.linear(x, w_q)

        # # Dynamically quantize activations and use real-valued weights.
        # with torch.no_grad():
        #     max_abs_x = x.detach().abs().amax()
        #     act_scale = (max_abs_x / self.qmax).clamp(min=torch.finfo(x.dtype).eps)
        # x_q = QuantFn.apply(x, act_scale.to(dtype=x.dtype, device=x.device), self.qmax)
        # y = F.linear(x_q, w_q)
        max_val = torch.max(x.abs().max(), y.abs().max())

        self.last_max_act = max_val.detach()
        return y

    def set_activation_scale(self, clip_value: float) -> None:
        with torch.no_grad():
            clip_value = torch.as_tensor(clip_value, dtype=self.log_act_s.dtype, device=self.log_act_s.device)
            scale = clip_value / self.qmax
            self.log_act_s.copy_(scale.log())

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
