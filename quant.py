import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from typing import Any, Dict


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


class QuantLinearWithWeights(nn.Module):
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

        # # Dynamically quantize activations and use real-valued weights.
        # with torch.no_grad():
        #     max_abs_x = x.detach().abs().amax()
        #     act_scale = (max_abs_x / self.qmax).clamp(min=torch.finfo(x.dtype).eps)
        # x_q = QuantFn.apply(x, act_scale.to(dtype=x.dtype, device=x.device), self.qmax)

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
        self.log_act_s.requires_grad = enabled
        self.log_weight_s.requires_grad = enabled


class QuantLinearWithScales(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
    ):
        super().__init__()
        self.qmax = 1 << (int(bits) - 1)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.log_act_s = nn.Parameter(torch.zeros(()))
        self.log_diag_s = nn.Parameter(torch.zeros(in_features))
        self.q_name: str | None = None
        self.last_max_act: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2-D tensor, got shape {tuple(x.shape)}")
        if self.log_diag_s.numel() != x.shape[-1]:
            raise ValueError(
                f"Incompatible SmoothQuant scales: {self.log_diag_s.numel()} vs hidden size {x.shape[-1]}"
            )
        inv = (-self.log_diag_s).exp().to(dtype=x.dtype, device=x.device)
        diag = self.log_diag_s.exp().to(dtype=self.weight.dtype, device=self.weight.device)
        view_shape = (1,) * (x.dim() - 1) + (inv.numel(),)
        x_scaled = x * inv.view(view_shape)
        act_scale = self.log_act_s.exp().to(dtype=x_scaled.dtype, device=x_scaled.device)
        x_q = QuantFn.apply(x_scaled, act_scale, self.qmax)
        weight_scaled = self.weight * diag.view(1, -1)
        y = F.linear(x_q, weight_scaled)
        with torch.no_grad():
            max_val = x_scaled.abs().max()
            self.last_max_act = max_val
        return y

    def set_scales(self, payload: Dict[str, Any]) -> None:
        scales = payload["scales"]
        act_percentile = payload["act_percentile"]
        scales = scales.to(device=self.weight.device, dtype=self.weight.dtype)
        act_percentile = act_percentile.to(device=self.weight.device, dtype=self.weight.dtype)
        eps = torch.finfo(scales.dtype).eps
        safe_scales = torch.clamp(scales, min=eps)

        ratio = act_percentile / safe_scales
        clip_value = torch.quantile(ratio.reshape(-1).to(torch.float32), 0.995).item()
        with torch.no_grad():
            self.log_diag_s.copy_(safe_scales.log().to(device=self.log_diag_s.device, dtype=self.log_diag_s.dtype))
            scale = torch.as_tensor(clip_value / float(self.qmax), dtype=self.log_act_s.dtype, device=self.log_act_s.device)
            self.log_act_s.copy_(scale.clamp(min=torch.finfo(scale.dtype).eps).log())

    def set_trainable(self, enabled: bool) -> None:
        self.weight.requires_grad = enabled
        self.log_act_s.requires_grad = enabled
        self.log_diag_s.requires_grad = enabled







# keep this for dynamic quant to try later
# class DiagScalingLinear(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         bits: int = 8,
#     ):
#         super().__init__()
#         self.qmax = 1 << (int(bits) - 1)
#         self.weight = nn.Parameter(torch.empty(out_features, in_features))
#         self.log_diag = nn.Parameter(torch.zeros(in_features))
#         self.q_name: str | None = None
#         self.last_max_act: torch.Tensor | None = None

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() < 2:
#             raise ValueError(f"Expected at least 2-D tensor, got shape {tuple(x.shape)}")
#         if self.log_diag.numel() != x.shape[-1]:
#             raise ValueError(
#                 f"Incompatible SmoothQuant scales: {self.log_diag.numel()} vs hidden size {x.shape[-1]}"
#             )
#         inv = (-self.log_diag).exp().to(dtype=x.dtype, device=x.device)
#         diag = self.log_diag.exp().to(dtype=self.weight.dtype, device=self.weight.device)
#         view_shape = (1,) * (x.dim() - 1) + (inv.numel(),)
#         x_scaled = x * inv.view(view_shape)

#         with torch.no_grad():
#             # Per-token (row-wise) dynamic activation scaling.
#             # For x_scaled shaped [..., H], compute a separate scale for each token/row
#             # by taking an L2 norm estimate over the hidden dim.
#             max_abs = x_scaled.detach().to(torch.float32).norm(p=2, dim=-1, keepdim=True)
#             eps = torch.finfo(x_scaled.dtype).eps
#             act_scale = (max_abs / float(self.qmax)).to(dtype=x_scaled.dtype, device=x_scaled.device)
#             act_scale = torch.clamp(act_scale, min=eps)

#         class _QuantNoScaleGrad(torch.autograd.Function):
#             @staticmethod
#             def forward(ctx, x_in, s_in, qmax_in) -> torch.Tensor:
#                 y = x_in / s_in
#                 q = y.clamp(-qmax_in, qmax_in - 1).round()
#                 xhat = q * s_in
#                 ctx.save_for_backward(y)
#                 ctx.qmax = qmax_in
#                 return xhat

#             @staticmethod
#             def backward(ctx, grad_out):
#                 (y,) = ctx.saved_tensors
#                 qmax_in = ctx.qmax
#                 mask = (-qmax_in <= y) & (y <= qmax_in - 1)
#                 grad_x = grad_out * mask.to(grad_out.dtype)
#                 return grad_x, None, None

#         x_q = _QuantNoScaleGrad.apply(x_scaled, act_scale, self.qmax)
#         weight_scaled = self.weight * diag.view(1, -1)
#         y = F.linear(x_q, weight_scaled)
#         with torch.no_grad():
#             max_val = x_scaled.abs().max()
#             self.last_max_act = max_val
#         return y

#     def set_diag_scales(self, payload: Dict[str, Any]) -> None:
#         if scales.dim() != 1:
#             raise ValueError("SmoothQuant scales must be a 1-D tensor.")
#         if scales.numel() != self.weight.shape[1]:
#             raise ValueError(
#                 f"Scale length {scales.numel()} does not match in_features {self.weight.shape[1]}"
#             )
#         scales = scales.to(device=self.weight.device, dtype=self.weight.dtype)
#         eps = torch.finfo(scales.dtype).eps
#         safe_scales = torch.clamp(scales, min=eps)
#         with torch.no_grad():
#             self.log_diag.copy_(safe_scales.log().to(device=self.log_diag.device, dtype=self.log_diag.dtype))

#     def set_trainable(self, enabled: bool) -> None:
#         self.weight.requires_grad = enabled
#         self.log_diag.requires_grad = enabled
