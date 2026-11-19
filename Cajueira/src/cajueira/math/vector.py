"""Small, explicit vector operations backed by NumPy."""

from __future__ import annotations

import numpy as np


def _as_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite numbers")
    return vector


def dot_product(a, b) -> float:
    """Return the dot product of two equal-length vectors."""
    left = _as_vector(a, "a")
    right = _as_vector(b, "b")
    if left.shape != right.shape:
        raise ValueError("vectors must have the same length")
    return float(np.dot(left, right))


def vector_norm(vector, *, p: float = 2.0) -> float:
    """Return the p-norm of a vector for p >= 1."""
    values = _as_vector(vector, "vector")
    if not np.isscalar(p) or not np.isfinite(p) or p < 1:
        raise ValueError("p must be a finite number >= 1")
    # Scale first to avoid overflow for large finite vector components.
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale == 0.0:
        return 0.0
    return float(scale * np.linalg.norm(values / scale, ord=p))


def cosine_similarity(a, b) -> float:
    """Return cosine similarity; define similarity with a zero vector as 0."""
    left = _as_vector(a, "a")
    right = _as_vector(b, "b")
    if left.shape != right.shape:
        raise ValueError("vectors must have the same length")
    # Normalize each vector independently before the dot product. Directly
    # computing norms/products can overflow or underflow for finite inputs.
    left_scale = float(np.max(np.abs(left), initial=0.0))
    right_scale = float(np.max(np.abs(right), initial=0.0))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    left_scaled = left / left_scale
    right_scaled = right / right_scale
    left_norm = np.linalg.norm(left_scaled)
    right_norm = np.linalg.norm(right_scaled)
    similarity = np.dot(left_scaled / left_norm, right_scaled / right_norm)
    # Floating-point roundoff may move the mathematical range by a few ulps.
    return float(np.clip(similarity, -1.0, 1.0))
