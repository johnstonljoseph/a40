import typing

import torch
from transformers.models.olmo3 import Olmo3ForCausalLM, Olmo3Model

from a40.utils import WeightedRMSNorm


def replace_with_weighted_norm(module: torch.nn.Module, attr: str, gamma: torch.Tensor) -> None:
    original = getattr(module, attr)
    weighted = WeightedRMSNorm(original, gamma=gamma)
    setattr(module, attr, weighted)


def fuse_ln_linear(
    layernorm: torch.nn.Module,
    linear_layers: typing.Iterable[torch.nn.Linear],
    side: str = "input",
) -> None:
    """
    fuse the scaling in Layernorm into the adjacent linear blocks.
    """
    if side not in {"input", "output"}:
        raise ValueError(f"Unsupported fuse side '{side}'. Expected 'input' or 'output'.")

    for linear in linear_layers:
        W_ = linear.weight.data.double()
        gamma = layernorm.weight.double()
        if side == "input":
            fused = W_ * gamma
        else:  # side == "output"
            fused = (gamma.view(-1, 1) * W_)
        linear.weight.data = fused.to(linear.weight.dtype)

        if hasattr(layernorm, "bias"):
            raise NotImplementedError("Bias in layernorm not supported.")

    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    

def fuse_layer_norms(model: Olmo3ForCausalLM) -> None:
    for layer in model.model.layers:
        attn_gamma = layer.post_attention_layernorm.weight.detach().clone()
        fuse_ln_linear(
            layer.post_attention_layernorm,
            [
                layer.self_attn.o_proj,
            ],
            side="output",
        )
        replace_with_weighted_norm(layer, "post_attention_layernorm", attn_gamma)

        mlp_gamma = layer.post_feedforward_layernorm.weight.detach().clone()
        fuse_ln_linear(
            layer.post_feedforward_layernorm,
            [
                layer.mlp.down_proj,
            ],
            side="output",
        )
        replace_with_weighted_norm(layer, "post_feedforward_layernorm", mlp_gamma)

    fuse_ln_linear(model.model.norm, [model.lm_head], side="input")


