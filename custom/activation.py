import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers.models.olmo3 import Olmo3Model


class _IdActivationFn(Function):
    @staticmethod
    def forward(
        ctx,
        blend: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(blend, gate)
        silu = F.silu(gate)

        return silu * (1.0 - blend) + gate * blend

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor]:
        (blend, gate) = ctx.saved_tensors
        blend = blend.to(dtype=gate.dtype)

        sig = torch.sigmoid(gate)
        dsilu = sig + gate * sig * (1.0 - sig)

        slope = torch.ones_like(gate)

        dact_dgate = dsilu * (1.0 - blend) + slope * blend
        grad_gate = grad_output * dact_dgate

        return None, grad_gate


class IdentityActivation(nn.Module):
    """Blend between SiLU (0.0) and learnable piecewise linear (1.0)."""
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.register_buffer("_blend", torch.tensor(0.0))

    def set_blend(self, value: float) -> None:
        self._blend.data.fill_(float(value))

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        gate = gate.to(self._blend.dtype)
        blend = self._blend.to(gate.dtype)
        return _IdActivationFn.apply(blend, gate)


class _PiecewiseActivationFn(Function):
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
        blend = blend.to(dtype=gate.dtype)

        sig = torch.sigmoid(gate)
        dsilu = sig + gate * sig * (1.0 - sig)

        t = gate - x0
        d = torch.as_tensor(ctx.delta, dtype=gate.dtype, device=gate.device)

        left = (t <= -d).to(dtype=gate.dtype)
        right = (t >= d).to(dtype=gate.dtype)
        mid = 1.0 - left - right

        slope_mid = ((a_p - a_m) / (2.0 * d)) * t + (a_p + a_m) / 2.0
        slope = left * a_m + right * a_p + mid * slope_mid

        dact_dgate = dsilu * (1.0 - blend) + slope * blend
        grad_gate = grad_output * dact_dgate

        weight_right = right + mid * (t + d) / (2.0 * d)
        weight_left = left + mid * (d - t) / (2.0 * d)

        grad_a_p = grad_output * blend * weight_right * t
        grad_a_m = grad_output * blend * weight_left * t
        grad_x0 = grad_output * blend * (-(a_p * weight_right + a_m * weight_left))
        grad_y0 = grad_output * blend * (-1.0)

        grad_a_p = grad_a_p.sum(dim=tuple(range(grad_a_p.dim() - 1)))
        grad_a_m = grad_a_m.sum(dim=tuple(range(grad_a_m.dim() - 1)))
        grad_x0 = grad_x0.sum()
        grad_y0 = grad_y0.sum()

        return None, None, None, grad_gate, grad_a_p, grad_a_m, grad_x0, grad_y0


class PiecewiseActivation(nn.Module):
    """Blend between SiLU (0.0) and learnable piecewise linear (1.0)."""

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
        return _PiecewiseActivationFn.apply(self._k, self._delta, blend, gate, a_p, a_m, x0, y0)


# def patch_layer_activation_and_params(
#     model: Olmo3Model, layer_index: int
# ) -> tuple[PiecewiseActivation, list[nn.Parameter]]:
#     mlp = _get_mlp(model, layer_index)
#     act_fn = PiecewiseActivation().to(
#         device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype
#     )
#     mlp.act_fn = act_fn

#     params: list[nn.Parameter] = list(mlp.parameters())
#     for p in params:
#         p.requires_grad = True

#     ln = getattr(model.layers[layer_index], "post_feedforward_layernorm")
#     weight = getattr(ln, "weight")
#     weight.requires_grad = True
#     params.append(weight)

#     return act_fn, params
