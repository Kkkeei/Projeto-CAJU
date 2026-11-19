"""TF-IDF text vectorizer."""

from __future__ import annotations

import numpy as np

from cajueira.nlp.tokenizer import tokenize
from cajueira.nlp.vocabulary import Vocabulary


class TfidfVectorizer:
    """Compute TF-IDF using smoothed inverse document frequency.

    IDF formula: log((1 + n_documents) / (1 + document_frequency)) + 1.
    """

    def __init__(
        self,
        *,
        sublinear_tf: bool = False,
        norm: str | None = "l2",
        min_frequency: int = 1,
    ) -> None:
        if norm not in {None, "l2"}:
            raise ValueError("norm must be None or 'l2'")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.min_frequency = min_frequency
        self.vocabulary_: Vocabulary | None = None
        self.idf_: np.ndarray | None = None

    def fit(self, texts: list[str]) -> "TfidfVectorizer":
        if not texts:
            raise ValueError("texts must contain at least one document")
        tokenized = [tokenize(text) for text in texts]
        self.vocabulary_ = Vocabulary(
            unk_token=None,
            min_frequency=self.min_frequency,
        ).fit_tokens(tokenized)
        document_frequency = np.zeros(len(self.vocabulary_), dtype=np.float64)
        for tokens in tokenized:
            for token in set(tokens):
                index = self.vocabulary_.word_to_index.get(token)
                if index is not None:
                    document_frequency[index] += 1
        n_documents = len(texts)
        self.idf_ = np.log((1.0 + n_documents) / (1.0 + document_frequency)) + 1.0
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.vocabulary_ is None or self.idf_ is None:
            raise RuntimeError("vectorizer is not fitted; call fit first")
        matrix = np.zeros((len(texts), len(self.vocabulary_)), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                index = self.vocabulary_.word_to_index.get(token)
                if index is not None:
                    matrix[row, index] += 1.0
        if self.sublinear_tf:
            nonzero = matrix > 0
            matrix[nonzero] = 1.0 + np.log(matrix[nonzero])
        matrix *= self.idf_
        if self.norm == "l2" and matrix.shape[1] > 0:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix /= norms
        return matrix

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    def get_feature_names_out(self) -> list[str]:
        if self.vocabulary_ is None:
            raise RuntimeError("vectorizer is not fitted")
        return list(self.vocabulary_.index_to_word)
