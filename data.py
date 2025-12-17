
from __future__ import annotations

import datasets
import random
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def build_dataloader(
    cfg: "Config",
    tokenizer_path: str,
    seed: int = 0,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    if rank == 0:
        print(f"[data] loading tokenizer from {tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    def make_transform(source: str):
        def tokenize_example(example: dict[str, Any]) -> dict[str, torch.Tensor]:
            if source == "SFT":
                messages = example["messages"]

            elif source == "DPO":
                messages = example["chosen"]

            elif source == "RL":
                outputs = example["outputs"]
                messages = [
                    {"role": "user", "content": example.get("prompt", "")},
                    {"role": "assistant", "content": random.choice(outputs)},
                ]

            else:
                raise ValueError(source)

            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                padding="max_length",
                truncation=True,
                max_length=cfg.seq_len,
                return_tensors="pt",
                return_dict=True,
            )

            return {
                "input_ids": encoded["input_ids"].squeeze(0).tolist(),
                "attention_mask": encoded["attention_mask"].squeeze(0).tolist(),
            }

        return tokenize_example

    datasets_to_mix = []
    probabilities = []

    def prepare_dataset(dataset_name: str, label: str):
        ds = datasets.load_dataset(
            dataset_name,
            split="train",
            streaming=True,
        )
        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
        if label == "RL":
            ds = ds.filter(lambda ex: bool(ex.get("outputs")))
        remove_columns = None
        if getattr(ds, "features", None) is not None:
            remove_columns = list(ds.features.keys())
        return ds.map(make_transform(label), remove_columns=remove_columns)

    if cfg.dataset_sft:
        ds_a = prepare_dataset(cfg.dataset_sft, "SFT")
        if rank == 0:
            print(f"[data] SFT dataset ready", flush=True)
        datasets_to_mix.append(ds_a)
        probabilities.append(cfg.dataset_ratio_sft)

    if cfg.dataset_dpo:
        ds_b = prepare_dataset(cfg.dataset_dpo, "DPO")
        if rank == 0:
            print(f"[data] DPO dataset ready", flush=True)
        datasets_to_mix.append(ds_b)
        probabilities.append(cfg.dataset_ratio_dpo)

    if cfg.dataset_rl:
        ds_c = prepare_dataset(cfg.dataset_rl, "RL")
        if rank == 0:
            print(f"[data] RL dataset ready", flush=True)
        datasets_to_mix.append(ds_c)
        probabilities.append(cfg.dataset_ratio_rl)

    if not datasets_to_mix:
        raise ValueError("No datasets configured.")

    total = sum(probabilities)
    mix = [p / total for p in probabilities]

    stream = datasets.interleave_datasets(
        datasets_to_mix,
        probabilities=mix,
        seed=seed,
        stopping_strategy="first_exhausted",
    )
    stream = stream.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=seed)
    stream = stream.with_format("torch")

    if rank == 0:
        print("[data] dataloader ready", flush=True)

    return DataLoader(
        stream,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
