"""Bag-of-Words count vectorizer."""

from __future__ import annotations

import numpy as np

from cajueira.nlp.vocabulary import Vocabulary
from cajueira.nlp.tokenizer import tokenize


class CountVectorizer:
    """Convert a collection of texts into token-count matrices."""

    def __init__(self, *, binary: bool = False, min_frequency: int = 1) -> None:
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        self.binary = binary
        self.min_frequency = min_frequency
        self.vocabulary_: Vocabulary | None = None

    def fit(self, texts: list[str]) -> "CountVectorizer":
        if not texts:
            raise ValueError("texts must contain at least one document")
        self.vocabulary_ = Vocabulary(
            unk_token=None,
            min_frequency=self.min_frequency,
        ).fit(texts)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.vocabulary_ is None:
            raise RuntimeError("vectorizer is not fitted; call fit first")
        matrix = np.zeros((len(texts), len(self.vocabulary_)), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                index = self.vocabulary_.word_to_index.get(token)
                if index is not None:
                    matrix[row, index] += 1.0
        if self.binary:
            matrix = (matrix > 0).astype(np.float64)
        return matrix

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    def get_feature_names_out(self) -> list[str]:
        if self.vocabulary_ is None:
            raise RuntimeError("vectorizer is not fitted")
        return list(self.vocabulary_.index_to_word)
