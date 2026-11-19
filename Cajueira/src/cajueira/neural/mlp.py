"""A small fully connected multiclass MLP implemented with NumPy."""

from __future__ import annotations

import numpy as np


class MLPClassifier:
    """Multilayer perceptron for tabular/vectorized classification.

    Hidden layers use ReLU and the output layer uses softmax. Training uses
    mini-batches and either SGD or Adam. This implementation is intentionally
    small and educational; it is not optimized for large datasets.
    """

    def __init__(
        self,
        *,
        hidden_sizes: tuple[int, ...] = (32,),
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer: str = "adam",
        random_state: int | None = 42,
        l2: float = 0.0,
    ) -> None:
        if (
            not isinstance(hidden_sizes, (tuple, list))
            or not hidden_sizes
            or any(
                isinstance(size, (bool, np.bool_))
                or not isinstance(size, (int, np.integer))
                or size < 1
                for size in hidden_sizes
            )
        ):
            raise ValueError("hidden_sizes must contain positive integers")
        for name, value in (
            ("learning_rate", learning_rate),
            ("l2", l2),
        ):
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name, value in (("epochs", epochs), ("batch_size", batch_size)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
        if learning_rate <= 0 or epochs < 1 or batch_size < 1 or l2 < 0:
            raise ValueError("invalid training hyperparameters")
        if optimizer not in {"sgd", "adam"}:
            raise ValueError("optimizer must be 'sgd' or 'adam'")
        self.hidden_sizes = tuple(hidden_sizes)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.optimizer = optimizer
        self.random_state = random_state
        self.l2 = float(l2)
        self.classes_: np.ndarray | None = None
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.loss_history_: list[float] = []

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _forward(self, X: np.ndarray):
        activations = [X]
        pre_activations = []
        current = X
        for weight, bias in zip(self.weights_[:-1], self.biases_[:-1]):
            z = current @ weight + bias
            pre_activations.append(z)
            current = np.maximum(0.0, z)
            activations.append(current)
        logits = current @ self.weights_[-1] + self.biases_[-1]
        probabilities = self._softmax(logits)
        return activations, pre_activations, probabilities

    def fit(self, X, y) -> "MLPClassifier":
        matrix = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
            raise ValueError("X must be a 2D matrix with at least two rows and one feature")
        if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
            raise ValueError("y must be a 1D array with one label per row")
        if labels.size == 0 or any(label is None for label in labels):
            raise ValueError("y must contain non-null labels")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("X must contain only finite values")
        self.classes_, encoded = np.unique(labels, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("at least two distinct classes are required")

        rng = np.random.default_rng(self.random_state)
        dimensions = [matrix.shape[1], *self.hidden_sizes, len(self.classes_)]
        self.weights_ = [
            rng.normal(0.0, np.sqrt(2.0 / dimensions[i]), (dimensions[i], dimensions[i + 1]))
            for i in range(len(dimensions) - 1)
        ]
        self.biases_ = [np.zeros((1, dimensions[i + 1])) for i in range(len(dimensions) - 1)]
        targets = np.eye(len(self.classes_))[encoded]
        self.loss_history_ = []

        first_m_w = [np.zeros_like(w) for w in self.weights_]
        second_m_w = [np.zeros_like(w) for w in self.weights_]
        first_m_b = [np.zeros_like(b) for b in self.biases_]
        second_m_b = [np.zeros_like(b) for b in self.biases_]
        step = 0
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        for _epoch in range(self.epochs):
            order = rng.permutation(matrix.shape[0])
            for start in range(0, matrix.shape[0], self.batch_size):
                batch_indices = order[start : start + self.batch_size]
                batch_X = matrix[batch_indices]
                batch_y = targets[batch_indices]
                activations, pre_activations, probabilities = self._forward(batch_X)
                batch_n = len(batch_indices)

                delta = (probabilities - batch_y) / batch_n
                grad_w = [None] * len(self.weights_)
                grad_b = [None] * len(self.biases_)
                grad_w[-1] = activations[-1].T @ delta + self.l2 * self.weights_[-1]
                grad_b[-1] = delta.sum(axis=0, keepdims=True)
                for layer in range(len(self.weights_) - 2, -1, -1):
                    delta = (delta @ self.weights_[layer + 1].T) * (pre_activations[layer] > 0)
                    grad_w[layer] = activations[layer].T @ delta + self.l2 * self.weights_[layer]
                    grad_b[layer] = delta.sum(axis=0, keepdims=True)

                step += 1
                for layer in range(len(self.weights_)):
                    if self.optimizer == "sgd":
                        self.weights_[layer] -= self.learning_rate * grad_w[layer]
                        self.biases_[layer] -= self.learning_rate * grad_b[layer]
                    else:
                        first_m_w[layer] = beta1 * first_m_w[layer] + (1 - beta1) * grad_w[layer]
                        second_m_w[layer] = beta2 * second_m_w[layer] + (1 - beta2) * (grad_w[layer] ** 2)
                        first_m_b[layer] = beta1 * first_m_b[layer] + (1 - beta1) * grad_b[layer]
                        second_m_b[layer] = beta2 * second_m_b[layer] + (1 - beta2) * (grad_b[layer] ** 2)
                        corrected_m_w = first_m_w[layer] / (1 - beta1 ** step)
                        corrected_v_w = second_m_w[layer] / (1 - beta2 ** step)
                        corrected_m_b = first_m_b[layer] / (1 - beta1 ** step)
                        corrected_v_b = second_m_b[layer] / (1 - beta2 ** step)
                        self.weights_[layer] -= self.learning_rate * corrected_m_w / (np.sqrt(corrected_v_w) + epsilon)
                        self.biases_[layer] -= self.learning_rate * corrected_m_b / (np.sqrt(corrected_v_b) + epsilon)

            _, _, epoch_probabilities = self._forward(matrix)
            clipped = np.clip(epoch_probabilities, 1e-12, 1.0)
            cross_entropy = -np.mean(np.sum(targets * np.log(clipped), axis=1))
            regularization = 0.5 * self.l2 * sum(np.sum(w * w) for w in self.weights_)
            self.loss_history_.append(float(cross_entropy + regularization))
        return self

    def _check_fitted(self) -> None:
        if self.classes_ is None or not self.weights_:
            raise RuntimeError("MLPClassifier is not fitted")

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.weights_[0].shape[0]:
            raise ValueError("X has an incompatible shape")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("X must contain only finite values")
        return self._forward(matrix)[2]

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X, y) -> float:
        labels = np.asarray(y)
        predictions = self.predict(X)
        if labels.ndim != 1 or len(labels) != len(predictions):
            raise ValueError("y must have one label per row in X")
        if labels.size == 0:
            raise ValueError("score requires at least one sample")
        return float(np.mean(predictions == labels))
