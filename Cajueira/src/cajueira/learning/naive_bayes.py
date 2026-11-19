"""Multinomial Naive Bayes classifier for non-negative features."""

from __future__ import annotations

import numpy as np


class MultinomialNaiveBayes:
    """A compact Multinomial Naive Bayes classifier.

    Suitable for count features and, in many practical cases, non-negative
    TF-IDF features. It is a probabilistic baseline, not a language generator.
    """

    def __init__(self, *, alpha: float = 1.0) -> None:
        if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
            raise TypeError("alpha must be a real number")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and > 0")
        self.alpha = float(alpha)
        self.classes_: np.ndarray | None = None
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None

    def fit(self, X, y) -> "MultinomialNaiveBayes":
        matrix = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D matrix")
        if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
            raise ValueError("y must be a 1D array with one label per row")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("X must contain finite, non-negative values")
        self.classes_, inverse = np.unique(labels, return_inverse=True)
        class_count = len(self.classes_)
        feature_count = matrix.shape[1]
        class_totals = np.bincount(inverse, minlength=class_count).astype(np.float64)
        feature_counts = np.zeros((class_count, feature_count), dtype=np.float64)
        for class_index in range(class_count):
            feature_counts[class_index] = matrix[inverse == class_index].sum(axis=0)
        smoothed = feature_counts + self.alpha
        self.feature_log_prob_ = np.log(
            smoothed / smoothed.sum(axis=1, keepdims=True)
        )
        self.class_log_prior_ = np.log(class_totals / class_totals.sum())
        return self

    def _check_fitted(self) -> None:
        if self.classes_ is None or self.class_log_prior_ is None or self.feature_log_prob_ is None:
            raise RuntimeError("classifier is not fitted")

    def predict_log_proba(self, X) -> np.ndarray:
        self._check_fitted()
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError("X has an incompatible shape")
        if matrix.shape[0] == 0:
            return np.empty((0, len(self.classes_)), dtype=np.float64)
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("X must contain finite, non-negative values")
        scores = matrix @ self.feature_log_prob_.T + self.class_log_prior_
        maximum = scores.max(axis=1, keepdims=True)
        logsumexp = maximum + np.log(np.exp(scores - maximum).sum(axis=1, keepdims=True))
        return scores - logsumexp

    def predict_proba(self, X) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        probabilities = self.predict_log_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
