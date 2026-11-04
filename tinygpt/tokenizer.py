"""Character-level tokenizer: one token per character.

Real LLMs use sub-word tokenizers (BPE, SentencePiece), but the testable
contract is the same: encode/decode must round-trip, the vocabulary must be
stable, and unknown input must be handled explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        if len(set(chars)) != len(chars):
            raise ValueError("vocabulary contains duplicate characters")
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        # sorted() makes the vocabulary independent of character order in the text
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        unknown = sorted({c for c in text if c not in self.stoi})
        if unknown:
            raise ValueError(f"characters not in vocabulary: {unknown}")
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["chars"])
