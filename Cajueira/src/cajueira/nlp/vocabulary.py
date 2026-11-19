"""Trainable vocabulary mapping tokens to stable integer IDs."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable, Sequence
import numbers

from cajueira.nlp.tokenizer import tokenize


class Vocabulary:
    """Build and manage a token-to-index mapping.

    The vocabulary is learned from a corpus; it is not a complete dictionary
    of a language. By default, ``<UNK>`` is reserved at index zero and is used
    for tokens that were not observed during fitting.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        unk_token: str | None = "<UNK>",
        pad_token: str | None = None,
        min_frequency: int = 1,
        max_size: int | None = None,
    ) -> None:
        for name, value in (("unk_token", unk_token), ("pad_token", pad_token)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")
        if isinstance(min_frequency, bool) or not isinstance(min_frequency, int):
            raise TypeError("min_frequency must be an integer")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        if max_size is not None:
            if isinstance(max_size, bool) or not isinstance(max_size, int):
                raise TypeError("max_size must be an integer or None")
            if max_size < 1:
                raise ValueError("max_size must be at least 1 or None")
        if unk_token is not None and pad_token == unk_token:
            raise ValueError("unk_token and pad_token must be different")

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.min_frequency = min_frequency
        self.max_size = max_size
        self.word_to_index: dict[str, int] = {}
        self.index_to_word: list[str] = []
        self.fitted = False
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        for token in (self.pad_token, self.unk_token):
            if token is not None and token not in self.word_to_index:
                self.word_to_index[token] = len(self.index_to_word)
                self.index_to_word.append(token)

    def fit(self, texts: Iterable[str]) -> "Vocabulary":
        """Learn tokens from texts, preserving first-seen order."""
        tokenized: list[list[str]] = [tokenize(text) for text in texts]
        return self.fit_tokens(tokenized)

    def fit_tokens(self, token_sequences: Iterable[Sequence[str]]) -> "Vocabulary":
        """Learn from already-tokenized sequences."""
        counts: Counter[str] = Counter()
        first_seen: dict[str, int] = {}
        position = 0
        for sequence in token_sequences:
            for token in sequence:
                if not isinstance(token, str):
                    raise TypeError("every token must be a string")
                if token not in first_seen:
                    first_seen[token] = position
                    position += 1
                counts[token] += 1

        reserved = {token for token in (self.unk_token, self.pad_token) if token}
        candidates = [
            token for token, count in counts.items()
            if count >= self.min_frequency and token not in reserved
        ]
        candidates.sort(key=lambda token: first_seen[token])
        if self.max_size is not None:
            candidates = candidates[: self.max_size]

        self.word_to_index = {}
        self.index_to_word = []
        self._add_special_tokens()
        for token in candidates:
            self.word_to_index[token] = len(self.index_to_word)
            self.index_to_word.append(token)
        self.fitted = True
        return self

    def add_token(self, token: str) -> int:
        """Add one token explicitly and return its index."""
        if not isinstance(token, str) or not token:
            raise ValueError("token must be a non-empty string")
        if token not in self.word_to_index:
            self.word_to_index[token] = len(self.index_to_word)
            self.index_to_word.append(token)
        return self.word_to_index[token]

    def encode(
        self,
        text: str,
        *,
        unknown: str = "unk",
    ) -> list[int]:
        """Convert text into token IDs.

        ``unknown`` may be ``'unk'``, ``'ignore'`` or ``'error'``.
        """
        if unknown not in {"unk", "ignore", "error"}:
            raise ValueError("unknown must be 'unk', 'ignore' or 'error'")
        ids: list[int] = []
        for token in tokenize(text):
            index = self.word_to_index.get(token)
            if index is not None:
                ids.append(index)
            elif unknown == "unk" and self.unk_token is not None:
                ids.append(self.word_to_index[self.unk_token])
            elif unknown == "error":
                raise KeyError(f"unknown token: {token!r}")
        return ids

    def decode(
        self,
        indices: Iterable[int],
        *,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """Convert token IDs back to tokens."""
        special = {token for token in (self.unk_token, self.pad_token) if token}
        decoded: list[str] = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, numbers.Integral):
                raise TypeError("every index must be an integer")
            index = int(index)
            if index < 0 or index >= len(self.index_to_word):
                raise IndexError(f"vocabulary index out of range: {index}")
            token = self.index_to_word[index]
            if skip_special_tokens and token in special:
                continue
            decoded.append(token)
        return decoded

    def __contains__(self, token: str) -> bool:
        return token in self.word_to_index

    def __len__(self) -> int:
        return len(self.index_to_word)

    def __getitem__(self, token: str) -> int:
        return self.word_to_index[token]

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "format_version": self.FORMAT_VERSION,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "min_frequency": self.min_frequency,
            "max_size": self.max_size,
            "tokens": self.index_to_word,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported vocabulary format version")
        vocab = cls(
            unk_token=data.get("unk_token"),
            pad_token=data.get("pad_token"),
            min_frequency=data.get("min_frequency", 1),
            max_size=data.get("max_size"),
        )
        tokens = data.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError("invalid vocabulary tokens")
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocabulary contains duplicate tokens")
        vocab.index_to_word = list(tokens)
        vocab.word_to_index = {token: index for index, token in enumerate(tokens)}
        expected = [t for t in (vocab.pad_token, vocab.unk_token) if t is not None]
        if any(token not in vocab.word_to_index for token in expected):
            raise ValueError("special tokens are missing from serialized vocabulary")
        vocab.fitted = True
        return vocab

    def save(self, path: str | Path) -> None:
        """Save vocabulary as UTF-8 JSON."""
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load a vocabulary saved with :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
