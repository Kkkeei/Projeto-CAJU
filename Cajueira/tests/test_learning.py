import numpy as np
import pytest

from cajueira.learning import MultinomialNaiveBayes


def test_naive_bayes_learns_simple_text_features():
    X = np.array([[3, 0], [2, 0], [0, 3], [0, 2]], dtype=float)
    y = np.array(["fruta", "fruta", "veiculo", "veiculo"])
    model = MultinomialNaiveBayes().fit(X, y)
    predictions = model.predict([[4, 0], [0, 4]])
    assert predictions.tolist() == ["fruta", "veiculo"]


def test_naive_bayes_probabilities_sum_to_one():
    X = np.array([[2, 0], [0, 2]], dtype=float)
    y = np.array([0, 1])
    probabilities = MultinomialNaiveBayes().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])


def test_naive_bayes_rejects_negative_features():
    with pytest.raises(ValueError):
        MultinomialNaiveBayes().fit([[1, -1], [0, 1]], [0, 1])


def test_naive_bayes_rejects_non_finite_alpha():
    with pytest.raises(ValueError):
        MultinomialNaiveBayes(alpha=float("nan"))
    with pytest.raises(ValueError):
        MultinomialNaiveBayes(alpha=float("inf"))


def test_naive_bayes_predict_empty_batch():
    model = MultinomialNaiveBayes().fit([[1.0], [2.0]], ["a", "b"])
    assert model.predict_proba(np.empty((0, 1))).shape == (0, 2)
