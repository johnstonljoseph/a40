import argparse
from pathlib import Path

import torch

from a40.utils import load_model, resolve_model_path
from .fuse_ln import fuse_layer_norms
from .rotate import rotate_model
from a40.custom_model import MyOlmo3ForCausalLM

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse LayerNorm scales into adjacent linear layers."
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="/workspace/.hf_home/hub/models--allenai--Olmo-3-7B-Think",
        help="Base model path (HF snapshot or local directory).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/workspace/a40/checkpoints/rotated",
        help="Directory to save the fused model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for loading and processing (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype to load the model with (e.g., float32, bfloat16).",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    base_path = resolve_model_path(args.base_path)
    model = load_model(base_path, device, dtype)

    fuse_layer_norms(model)
    # rotate_model(model, device)

    fused_model = MyOlmo3ForCausalLM(model.config)
    fused_model.load_state_dict(model.state_dict())

    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fused_model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
