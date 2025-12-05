from __future__ import annotations

from typing import Any, Dict, Iterator

import datasets
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import LlamaTokenizerFast

if False:  # type-checking placeholder without runtime import
    from main import Config  # pragma: no cover

class PackedStreamingDataset(IterableDataset):
    """
    Concatenate all tokens from the stream and yield dense seq_len chunks.
    No padding, no truncation waste—every token is used.
    """

    def __init__(self, stream, tokenizer_path: str, seq_len: int) -> None:
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
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


def format_to_text(example: Dict[str, Any], tokenizer: LlamaTokenizerFast) -> Dict[str, str]:
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


def has_text(example: Dict[str, Any]) -> bool:
    return bool(example.get("text").strip())


def build_dataloader(cfg: "Config", tokenizer_path: str) -> DataLoader:
    template_tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path, use_fast=True)

    def map_fn(example):
        return format_to_text(example, template_tokenizer)

    ds_a = (
        datasets.load_dataset(cfg.dataset_a, split="train", streaming=True)
        .map(map_fn)
        .filter(has_text)
        .repeat()
    )
    ds_b = (
        datasets.load_dataset(cfg.dataset_b, split="train", streaming=True)
        .map(map_fn)
        .filter(has_text)
        .repeat()
    )

    stream = datasets.interleave_datasets(
        [ds_a, ds_b],
        probabilities=[cfg.dataset_ratio_a, 1.0 - cfg.dataset_ratio_a],
        seed=cfg.seed,
    ).shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)

    iterable = PackedStreamingDataset(stream, tokenizer_path, cfg.seq_len)
    return DataLoader(
        iterable,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
