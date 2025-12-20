import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers.models.olmo3 import Olmo3Model


# Silu

class _Silu(Function):
    @staticmethod
    def forward(
        ctx,
        blend: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(gate)
        silu = F.silu(gate)

        return silu * (1.0 - blend) + gate * blend

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor]:
        gate = ctx.saved_tensors[0]

        sig = torch.sigmoid(gate)
        dsilu = sig + gate * sig * (1.0 - sig)

        grad_gate = grad_output * dsilu

        return None, grad_gate


class SiluActivation(nn.Module):
    """SiLU with optional blend toward identity (for interface compatibility)."""

    def __init__(self) -> None:
        super().__init__()
        # Non-persistent so checkpoints cannot override runtime-enforced blend.
        self.register_buffer("_blend", torch.tensor(0.0), persistent=False)

    def set_blend(self, value: float) -> None:
        self._blend.data.fill_(float(value))

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        gate = gate.to(self._blend.dtype)
        blend = self._blend.to(gate.dtype)
        return _Silu.apply(blend, gate)


# Id

class IdentityActivation(nn.Module):
    """Blend between SiLU (0.0) and learnable piecewise linear (1.0)."""
    def __init__(
        self,
    ) -> None:
        super().__init__()
        # Non-persistent so checkpoints cannot override runtime-enforced blend.
        self.register_buffer("_blend", torch.tensor(0.0), persistent=False)

    def set_blend(self, value: float) -> None:
        self._blend.data.fill_(float(value))

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        return gate


class _IdActivationWithBlendFn(Function):
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


class IdentityWithBlendActivation(nn.Module):
    """Blend between SiLU (0.0) and learnable piecewise linear (1.0)."""
    def __init__(
        self,
    ) -> None:
        super().__init__()
        # Non-persistent so checkpoints cannot override runtime-enforced blend.
        self.register_buffer("_blend", torch.tensor(0.0), persistent=False)

    def set_blend(self, value: float) -> None:
        self._blend.data.fill_(float(value))

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        gate = gate.to(self._blend.dtype)
        blend = self._blend.to(gate.dtype)
        return _IdActivationWithBlendFn.apply(blend, gate)


# Piecewise

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


class PiecewiseActivationWithBlend(nn.Module):
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
        # Keep blend non-persistent so checkpoints cannot overwrite the enforced value.
        self.register_buffer("_blend", torch.tensor(1.0), persistent=False)
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


class PiecewiseActivation(nn.Module):
    """Always-on piecewise activation (equivalent to blend=1.0 branch)."""

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
        self._a_p = nn.Parameter(torch.tensor(float(a_p)))
        self._a_m = nn.Parameter(torch.tensor(float(a_m)))
        self._x0 = nn.Parameter(torch.tensor(float(x0)))
        self._y0 = nn.Parameter(torch.tensor(float(y0)))
        self._k = float(k)
        self._delta = float(delta)

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        a_p = self._a_p.to(dtype=gate.dtype)
        a_m = self._a_m.to(dtype=gate.dtype)
        x0 = self._x0.to(dtype=gate.dtype)
        y0 = self._y0.to(dtype=gate.dtype)
        # Blend is fixed at 1.0; we skip the SiLU branch entirely.
        return _PiecewiseActivationFn.apply(
            self._k,
            self._delta,
            torch.tensor(1.0, device=gate.device, dtype=gate.dtype),
            gate,
            a_p,
            a_m,
            x0,
            y0,
        )


# Offset Relu

class _OffsetReluFunction(Function):
    """
    Forward: blend between SiLU (1 - blend) and hard offset ReLU (blend) with fixed slopes (a_p=1, a_m=0),
    offset by x0/y0. Backward: smooth surrogate using sigmoid slope.
    """

    @staticmethod
    def forward(
        ctx,
        blend: torch.Tensor,
        gate: torch.Tensor,
        x0: torch.Tensor,
        y0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        blend = blend.to(dtype=gate.dtype)
        x0_tensor = x0.to(dtype=gate.dtype, device=gate.device)
        y0_tensor = y0.to(dtype=gate.dtype, device=gate.device)
        t_tensor = t.to(dtype=gate.dtype, device=gate.device)

        silu = F.silu(gate)
        g = gate - x0_tensor
        offset = torch.relu(g) - y0_tensor
        out = silu * (1.0 - blend) + offset * blend

        # Save for surrogate backward
        ctx.save_for_backward(blend, gate, x0_tensor, t_tensor)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        blend, gate, x0_tensor, t_tensor = ctx.saved_tensors
        blend = blend.to(dtype=gate.dtype)
        sig = torch.sigmoid(gate)
        dsilu = sig + gate * sig * (1.0 - sig)
        # Surrogate slope for the offset ReLU branch, centered at x0
        slope = torch.sigmoid((gate - x0_tensor) / t_tensor)
        dact_dgate = dsilu * (1.0 - blend) + slope * blend
        grad_gate = grad_output * dact_dgate
        return None, grad_gate, None, None, None


class OffsetReluActivation(nn.Module):
    """
    Fixed offset ReLU surrogate:
        f(gate) ≈ relu(gate - x0) - y0
    with a_p=1, a_m=0, x0=-0.2, y0=0.2; all frozen (no trainable params).
    """

    def __init__(self, x0: float = -0.2, y0: float = 0.2, t: float = 0.02) -> None:
        super().__init__()
        # store as buffers so they move with the module but are not trainable
        # Keep blend persistent=False so checkpoints don't overwrite our forced value.
        self.register_buffer("_blend", torch.tensor(1.0), persistent=False)
        self.register_buffer("_x0", torch.tensor(float(x0)))
        self.register_buffer("_y0", torch.tensor(float(y0)))
        self.register_buffer("_t", torch.tensor(float(t)))

    def set_blend(self, value: float) -> None:
        self._blend.data.fill_(float(value))

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        gate = gate.to(dtype=self._blend.dtype)
        blend = self._blend.to(gate.dtype)
        x0 = self._x0.to(dtype=gate.dtype, device=gate.device)
        y0 = self._y0.to(dtype=gate.dtype, device=gate.device)
        t = self._t.to(dtype=gate.dtype, device=gate.device)
        return _OffsetReluFunction.apply(
            blend,
            gate,
            x0,
            y0,
            t,
        )


# Leaky

class _LeakyActivationFn(Function):
    @staticmethod
    def forward(
        ctx,
        blend: torch.Tensor,
        gate: torch.Tensor,
        a_m: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(blend, gate, a_m)
        silu = F.silu(gate)

        leaky = torch.where(gate >= 0, gate, a_m * gate)
        return silu * (1.0 - blend) + leaky * blend

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor, torch.Tensor]:
        blend, gate, a_m = ctx.saved_tensors
        blend = blend.to(dtype=gate.dtype)

        sig = torch.sigmoid(gate)
        dsilu = sig + gate * sig * (1.0 - sig)

        t = gate.new_tensor(0.02)
        a = a_m
        sig_sur = torch.sigmoid(gate / t)
        df_dgate = a + (1.0 - a) * sig_sur

        dact_dgate = dsilu * (1.0 - blend) + df_dgate * blend
        grad_gate = grad_output * dact_dgate

        soft = F.softplus(gate / t)
        df_da = gate - (t * soft)
        grad_a_m = (grad_output * blend * df_da).sum()

        return None, grad_gate, grad_a_m


class LeakyActivation(nn.Module):
    """Blend between SiLU (0.0) and learnable leaky ReLU (1.0)."""

    def __init__(self, a_m: float = 0.05) -> None:
        super().__init__()
        self.register_buffer("_blend", torch.tensor(0.0))
        self._a_m = nn.Parameter(torch.tensor(float(a_m)))

    def set_blend(self, value: float) -> None:
        self._blend.data.fill_(float(value))

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        gate = gate.to(self._blend.dtype)
        blend = self._blend.to(gate.dtype)
        a_m = self._a_m.to(dtype=gate.dtype)
        return _LeakyActivationFn.apply(blend, gate, a_m)


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
