from __future__ import annotations

from typing import Dict, Iterator

import datasets
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

class PackedStreamingDataset(IterableDataset):
    """
    Concatenate all tokens from the stream and yield dense seq_len chunks.
    No padding, no truncation waste—every token is used.

    We also do manual sharding across DDP ranks: each rank only consumes
    every `world_size`-th document from the underlying stream.
    """

    def __init__(
        self,
        stream,
        tokenizer_path: str,
        seq_len: int,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
        )
        self.stream = stream
        self.seq_len = seq_len
        self.eos_id = self.tokenizer.eos_token_id
        self.world_size = world_size
        self.rank = rank

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer: list[int] = []
        doc_idx = 0
        
        for example in self.stream:
            if (doc_idx % self.world_size) != self.rank:
                doc_idx += 1
                continue
            doc_idx += 1

            # Tokenize without padding/truncation; get raw token ids
            ids = self.tokenizer.encode(example["text"], add_special_tokens=False)
            ids.append(self.eos_id)
            buffer.extend(ids)

            # Yield as many full chunks as possible
            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                }


def to_text_stream(ds, tokenizer):
    text_features = datasets.Features({"text": datasets.Value("string")})

    def generator():
        for example in ds:
            text = example.get("text")
            if isinstance(text, str) and text.strip():
                yield {"text": text}
            elif isinstance(messages := example.get("messages"), list) and messages:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                ).strip()
                if rendered:
                    yield {"text": rendered}

    return datasets.IterableDataset.from_generator(generator, features=text_features)


def build_dataloader(
    cfg: "Config",
    tokenizer_path: str,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    if rank == 0:
        print(f"[data] loading tokenizer from {tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    ds_a = to_text_stream(
        datasets.load_dataset(cfg.dataset_a, split="train", streaming=True),
        tokenizer,
    )
    if rank == 0:
        print(f"[data] dataset A ready", flush=True)
    ds_b = to_text_stream(
        datasets.load_dataset(cfg.dataset_b, split="train", streaming=True),
        tokenizer,
    )
    if rank == 0:
        print(f"[data] dataset B ready", flush=True)

    stream = datasets.interleave_datasets(
        [ds_a, ds_b],
        probabilities=[cfg.dataset_ratio_a, 1.0 - cfg.dataset_ratio_a],
        # [ds_a],
        # probabilities=[1.0],
        seed=cfg.seed,
    )
    stream = stream.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    stream = stream.repeat(None)

    if rank == 0:
        print("[data] packing tokens", flush=True)
    iterable = PackedStreamingDataset(
        stream,
        tokenizer_path,
        cfg.seq_len,
        world_size=world_size,
        rank=rank,
    )
    if rank == 0:
        print("[data] dataloader ready", flush=True)

    return DataLoader(
        iterable,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
