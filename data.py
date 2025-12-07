from __future__ import annotations

from typing import Any, Dict, Iterator, Iterable

import datasets
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

class PackedStreamingDataset(IterableDataset):
    """
    Concatenate all tokens from the stream and yield dense seq_len chunks.
    No padding, no truncation waste—every token is used.
    """

    def __init__(self, stream, tokenizer_path: str, seq_len: int) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
        )
        self.stream = stream
        self.seq_len = seq_len
        self.eos_id = self.tokenizer.eos_token_id

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer: list[int] = []

        for example in self.stream:
            # Tokenize without padding/truncation; get raw token ids
            ids = self.tokenizer.encode(example["text"], add_special_tokens=False)
            # Append EOS to mark document boundary
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


def format_to_text(example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> Dict[str, str]:
    text = example.get("text")
    if isinstance(text, str) and text.strip():
        return {"text": text}

    messages = example.get("messages")
    if isinstance(messages, list) and messages:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": rendered.strip()}

    return {"text": ""}


def _has_text(example: Dict[str, Any]) -> bool:
    text = example.get("text")
    return isinstance(text, str) and bool(text.strip())


def build_dataloader(
    cfg: "Config",
    tokenizer_path: str,
    *,
    num_shards: int = 1,
    shard_rank: int = 0,
) -> DataLoader:
    print(f"[data] loading tokenizer from {tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    ds_a = (
        datasets.load_dataset(cfg.dataset_a, split="train", streaming=True)
        .map(format_to_text, fn_kwargs={"tokenizer": tokenizer})
        .filter(_has_text)
        .repeat(None)
    )
    print(f"[data] dataset A ready", flush=True)
    # ds_b = (
    #     datasets.load_dataset(cfg.dataset_b, split="train", streaming=True)
    #     .map(format_to_text, fn_kwargs={"tokenizer": tokenizer})
    #     .filter(_has_text)
    #     .repeat(None)
    # )
    # print(f"[data] dataset B ready", flush=True)

    print("[data] interleaving + shuffling streams", flush=True)
    repeated_stream = datasets.interleave_datasets(
        # [ds_a, ds_b],
        # probabilities=[cfg.dataset_ratio_a, 1.0 - cfg.dataset_ratio_a],
        [ds_a],
        probabilities=[1.0],
        seed=cfg.seed,
    ).shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)

    if num_shards > 1:
        repeated_stream = repeated_stream.shard(num_shards=num_shards, index=shard_rank, contiguous=True)

    print("[data] packing tokens", flush=True)
    iterable = PackedStreamingDataset(repeated_stream, tokenizer_path, cfg.seq_len)
    print("[data] dataloader ready", flush=True)
    return DataLoader(
        iterable,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        # prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
