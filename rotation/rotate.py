
import torch
from typing import Optional
# from transformers import Olmo3ForCausalLM

from transformers.models.olmo3.modeling_olmo3 import (
    Olmo3DecoderLayer,
    Olmo3Attention,
    Olmo3MLP,
    Olmo3ForCausalLM
)


def random_orthogonal_matrix(
    dim: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a random orthogonal matrix using QR decomposition."""
    normal = torch.randn(dim, dim, device=device, dtype=torch.float64, generator=generator)
    q, r = torch.linalg.qr(normal)
    # To make q consistent across qr implementations, we must incorporate diagonal sign info from r into q
    diag = torch.diagonal(r)
    phases = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
    return q * phases.unsqueeze(0)


def rotate_embeddings(model: Olmo3ForCausalLM, Q: torch.Tensor, device: torch.device):
    W = model.model.embed_tokens
    W_ = W.weight.to(device=device, dtype=torch.float64)
    W.weight.copy_(torch.matmul(W_, Q).to(device="cpu", dtype=W.weight.dtype))


def rotate_head(model: Olmo3ForCausalLM, Q: torch.Tensor, device: torch.device):
    W = model.lm_head
    W_ = W.weight.to(device=device, dtype=torch.float64)
    W.weight.copy_(torch.matmul(W_, Q).to(device="cpu", dtype=W.weight.dtype))


def rotate_attention_io(
    layer: Olmo3DecoderLayer,
    Q: torch.Tensor,
    # R_query: torch.Tensor,
    # R_key: torch.Tensor,
    # R_value: torch.Tensor,
    # R_output: torch.Tensor,
    device: torch.device,
):
    attn: Olmo3Attention = layer.self_attn
    
    Wq = attn.q_proj
    Wq_ = Wq.weight.to(device=device, dtype=torch.float64)
    Wq_rot = torch.matmul(Wq_, Q)
    # Wq_rot = torch.matmul(R_query, Wq_rot)
    Wq.weight.copy_(Wq_rot.to(device="cpu", dtype=Wq.weight.dtype))

    Wk = attn.k_proj
    Wk_ = Wk.weight.to(device=device, dtype=torch.float64)
    Wk_rot = torch.matmul(Wk_, Q)
    # Wk_rot = torch.matmul(R_key, Wk_rot)
    Wk.weight.copy_(Wk_rot.to(device="cpu", dtype=Wk.weight.dtype))

    Wv = attn.v_proj
    Wv_ = Wv.weight.to(device=device, dtype=torch.float64)
    Wv_rot = torch.matmul(Wv_, Q)
    # Wv_rot = torch.matmul(R_value, Wv_rot)
    Wv.weight.copy_(Wv_rot.to(device="cpu", dtype=Wv.weight.dtype))

    Wo = attn.o_proj
    Wo_ = Wo.weight.to(device=device, dtype=torch.float64)
    # Wo_rot = torch.matmul(Wo_, R_output.T)
    Wo_rot = torch.matmul(Q.T, Wo_rot)
    Wo.weight.copy_(Wo_rot.to(device="cpu", dtype=Wo.weight.dtype))
    

def rotate_mlp_io(layer: Olmo3DecoderLayer, Q: torch.Tensor, device: torch.device):
    mlp: Olmo3MLP = layer.mlp

    for W in [mlp.up_proj, mlp.gate_proj]:
        W_ = W.weight.to(device=device, dtype=torch.float64)
        W.weight.copy_(torch.matmul(W_, Q).to(device="cpu", dtype=W.weight.dtype))
    
    for W in [mlp.down_proj]:
        W_ = W.weight.to(device=device, dtype=torch.float64)
        W.weight.copy_(torch.matmul(Q.T, W_).to(device="cpu", dtype=W.weight.dtype))


def rotate_model(model: Olmo3ForCausalLM, device: torch.device):
    Q = random_orthogonal_matrix(model.config.hidden_size, device)

    with torch.no_grad():
        rotate_embeddings(model, Q, device)
        rotate_head(model, Q, device)
        for index, layer in enumerate(model.model.layers):
            print(f"Rotating layer {index}/{len(model.model.layers)}...")
            rotate_attention_io(layer, Q, device)
            rotate_mlp_io(layer, Q, device)




