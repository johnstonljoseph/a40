import argparse
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from a40.utils import load_model, resolve_model_path
from a40.custom_model import MyOlmo3ForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two checkpoints by estimating KL divergence over random prompts."
    )
    parser.add_argument("--model-a", type=str, required=True, help="Path to the first model.")
    parser.add_argument("--model-b", type=str, required=True, help="Path to the second model.")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path (defaults to --model-a).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., cuda, cpu).")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype to load models with (e.g., float32, bfloat16).",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for random prompts.")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length of prompts.")
    parser.add_argument("--num-batches", type=int, default=4, help="Number of batches to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for prompt generation.")
    return parser.parse_args()


@torch.no_grad()
def compute_masked_kl(p_logits: torch.Tensor, q_logits: torch.Tensor, attention_mask: torch.Tensor) -> float:
    """KL(P || Q) averaged over unmasked tokens."""
    p = p_logits.log_softmax(dim=-1)
    q = q_logits.log_softmax(dim=-1)
    probs_p = p.exp()
    token_kl = (probs_p * (p - q)).sum(dim=-1)
    mask = attention_mask.to(dtype=token_kl.dtype)
    return (token_kl * mask).sum().div(mask.sum()).item()


def sample_inputs(
    tokenizer: AutoTokenizer,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for row in range(batch_size):
        length = rng.randint(max(1, seq_len // 4), seq_len)
        tokens = torch.randint(low=0, high=vocab_size, size=(length,), dtype=torch.long)
        input_ids[row, :length] = tokens
        attention_mask[row, :length] = 1

    return input_ids.to(device), attention_mask.to(device)


def load_fused_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> MyOlmo3ForCausalLM:
    model = MyOlmo3ForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
    )
    model.config.use_cache = False
    model.eval()
    model = model.to(device)
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    tokenizer_base = args.tokenizer_path or args.model_a
    tokenizer_path = resolve_model_path(tokenizer_base)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )

    model_a_path = resolve_model_path(args.model_a)
    model_b_path = resolve_model_path(args.model_b)
    print(f"[compare] loading model A from {model_a_path}", flush=True)
    model_a = load_model(model_a_path, device, dtype)
    print(f"[compare] loading model B from {model_b_path}", flush=True)
    model_b = load_fused_model(model_b_path, device, dtype)

    rng = random.Random(args.seed)
    kl_a_to_b = []
    kl_b_to_a = []

    for batch_idx in range(args.num_batches):
        input_ids, attention_mask = sample_inputs(
            tokenizer,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
            rng=rng,
        )

        with torch.no_grad():
            logits_a = model_a(input_ids=input_ids, attention_mask=attention_mask).logits
            logits_b = model_b(input_ids=input_ids, attention_mask=attention_mask).logits

        kl_ab = compute_masked_kl(logits_a, logits_b, attention_mask)
        kl_ba = compute_masked_kl(logits_b, logits_a, attention_mask)
        kl_a_to_b.append(kl_ab)
        kl_b_to_a.append(kl_ba)

        print(
            f"[batch {batch_idx}] KL(A||B)={kl_ab:.6f} KL(B||A)={kl_ba:.6f}",
            flush=True,
        )

    avg_ab = sum(kl_a_to_b) / len(kl_a_to_b)
    avg_ba = sum(kl_b_to_a) / len(kl_b_to_a)
    print(f"[summary] avg KL(A||B)={avg_ab:.8f}")
    print(f"[summary] avg KL(B||A)={avg_ba:.8f}")


if __name__ == "__main__":
    main()
