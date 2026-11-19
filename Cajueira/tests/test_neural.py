import numpy as np
import pytest

from cajueira.neural import MLPClassifier


def test_mlp_learns_two_separable_classes():
    X = np.array([
        [-2.0, -1.0], [-1.5, -2.0], [-1.0, -1.0],
        [1.0, 1.0], [1.5, 2.0], [2.0, 1.0],
    ])
    y = np.array(["A", "A", "A", "B", "B", "B"])
    model = MLPClassifier(
        hidden_sizes=(8,), epochs=180, learning_rate=0.03,
        batch_size=6, random_state=7,
    ).fit(X, y)
    assert model.score(X, y) >= 0.95
    assert len(model.loss_history_) == 180


def test_mlp_probabilities_sum_to_one():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array(["zero", "one", "one", "zero"])
    model = MLPClassifier(epochs=5, random_state=1).fit(X, y)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_mlp_requires_at_least_two_classes():
    with pytest.raises(ValueError):
        MLPClassifier(epochs=2).fit([[0], [1]], ["same", "same"])


def test_mlp_rejects_unknown_optimizer():
    with pytest.raises(ValueError):
        MLPClassifier(optimizer="rmsprop")


def test_mlp_supports_sgd_and_multiple_hidden_layers():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array(["low", "low", "high", "high"])
    model = MLPClassifier(
        hidden_sizes=(5, 3), epochs=8, batch_size=4,
        learning_rate=0.05, optimizer="sgd", random_state=3,
    ).fit(X, y)
    assert model.predict(X).shape == (4,)
    assert np.isfinite(model.loss_history_).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_sizes": (2.5,)},
        {"hidden_sizes": (True,)},
        {"epochs": True},
        {"batch_size": 2.5},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"l2": float("inf")},
        {"l2": float("nan")},
    ],
)
def test_mlp_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises((TypeError, ValueError)):
        MLPClassifier(**kwargs)


def test_mlp_score_rejects_empty_dataset():
    X = np.array([[0.0], [1.0]])
    y = np.array(["a", "b"])
    model = MLPClassifier(epochs=1).fit(X, y)
    with pytest.raises(ValueError):
        model.score(np.empty((0, 1)), np.array([]))


def test_mlp_backpropagation_matches_finite_differences():
    """Check analytical gradients against numerical central differences."""
    model = MLPClassifier(hidden_sizes=(3,), epochs=1, random_state=0, l2=0.07)
    X = np.array([[0.2, -0.4], [0.7, 0.1]])
    labels = np.array([0, 1])
    model.classes_ = np.array([0, 1])
    model.weights_ = [
        np.array([[0.1, 0.2, -0.3], [0.4, -0.2, 0.2]]),
        np.array([[0.2, -0.1], [0.1, 0.3], [-0.2, 0.4]]),
    ]
    model.biases_ = [
        np.array([[0.01, -0.02, 0.03]]),
        np.array([[0.02, -0.01]]),
    ]

    activations, pre_activations, probabilities = model._forward(X)
    targets = np.eye(2)[labels]
    delta = (probabilities - targets) / len(X)
    grad_weights = [None] * len(model.weights_)
    grad_biases = [None] * len(model.biases_)
    grad_weights[-1] = activations[-1].T @ delta + model.l2 * model.weights_[-1]
    grad_biases[-1] = delta.sum(axis=0, keepdims=True)
    for layer in range(len(model.weights_) - 2, -1, -1):
        delta = (delta @ model.weights_[layer + 1].T) * (pre_activations[layer] > 0)
        grad_weights[layer] = activations[layer].T @ delta + model.l2 * model.weights_[layer]
        grad_biases[layer] = delta.sum(axis=0, keepdims=True)

    def objective():
        probs = model._forward(X)[2]
        cross_entropy = -np.mean(
            np.sum(targets * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        )
        penalty = 0.5 * model.l2 * sum(np.sum(w * w) for w in model.weights_)
        return cross_entropy + penalty

    epsilon = 1e-6
    for parameter_group, gradient_group in (
        (model.weights_, grad_weights),
        (model.biases_, grad_biases),
    ):
        for parameter, analytical in zip(parameter_group, gradient_group):
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                loss_plus = objective()
                parameter[index] = original - epsilon
                loss_minus = objective()
                parameter[index] = original
                numerical = (loss_plus - loss_minus) / (2 * epsilon)
                assert analytical[index] == pytest.approx(numerical, abs=2e-7)
