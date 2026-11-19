"""A simple intent router based on TF-IDF centroids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cajueira.math.vector import cosine_similarity
from cajueira.vectors.tfidf import TfidfVectorizer


@dataclass(frozen=True)
class IntentResult:
    """Result of routing a message to an intent."""

    intent: str | None
    similarity: float
    scores: dict[str, float]


class IntentRouter:
    """Route a text to the closest intent using example-based TF-IDF.

    Similarity is a lexical score, not a calibrated probability or proof of
    semantic understanding. Add varied representative examples per intent.
    """

    def __init__(self, *, threshold: float = 0.15) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = float(threshold)
        self._examples: dict[str, list[str]] = {}
        self.vectorizer_: TfidfVectorizer | None = None
        self.centroids_: dict[str, np.ndarray] = {}

    def add_intent(self, intent: str, examples: list[str]) -> "IntentRouter":
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("intent must be a non-empty string")
        if not examples or any(not isinstance(item, str) or not item.strip() for item in examples):
            raise ValueError("examples must be a non-empty list of non-empty strings")
        self._examples.setdefault(intent, []).extend(examples)
        self.vectorizer_ = None
        self.centroids_ = {}
        return self

    def fit(self) -> "IntentRouter":
        if not self._examples:
            raise ValueError("add at least one intent before fitting")
        all_texts = [text for examples in self._examples.values() for text in examples]
        self.vectorizer_ = TfidfVectorizer().fit(all_texts)
        self.centroids_ = {}
        for intent, examples in self._examples.items():
            matrix = self.vectorizer_.transform(examples)
            centroid = matrix.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self.centroids_[intent] = centroid
        return self

    def route(self, text: str) -> IntentResult:
        if self.vectorizer_ is None or not self.centroids_:
            raise RuntimeError("router is not fitted; call fit first")
        vector = self.vectorizer_.transform([text])[0]
        scores = {
            intent: cosine_similarity(vector, centroid)
            for intent, centroid in self.centroids_.items()
        }
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        # An empty/OOV message has no lexical evidence. It must not be routed,
        # even when the configured threshold is zero.
        has_signal = bool(np.any(vector))
        accepted_intent = best_intent if has_signal and best_score >= self.threshold else None
        return IntentResult(
            intent=accepted_intent,
            similarity=float(best_score),
            scores=scores,
        )
