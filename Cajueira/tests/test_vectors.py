import numpy as np
import pytest

from cajueira.vectors import CountVectorizer, TfidfVectorizer


def test_count_vectorizer_counts_terms():
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(["gato gato cão", "cão"])
    assert vectorizer.get_feature_names_out() == ["gato", "cão"]
    np.testing.assert_array_equal(matrix, [[2, 1], [0, 1]])


def test_binary_count_vectorizer():
    matrix = CountVectorizer(binary=True).fit_transform(["a a b"])
    np.testing.assert_array_equal(matrix, [[1, 1]])


def test_count_vectorizer_ignores_unseen_tokens():
    vectorizer = CountVectorizer().fit(["casa carro"])
    matrix = vectorizer.transform(["casa avião"])
    np.testing.assert_array_equal(matrix, [[1, 0]])


def test_tfidf_shape_and_l2_normalization():
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(["gato cão", "gato peixe"])
    assert matrix.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0])


def test_tfidf_unseen_tokens_produce_zero_vector():
    vectorizer = TfidfVectorizer().fit(["casa carro"])
    matrix = vectorizer.transform(["avião foguete"])
    np.testing.assert_array_equal(matrix, [[0.0, 0.0]])


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        CountVectorizer().transform(["texto"])
    with pytest.raises(RuntimeError):
        TfidfVectorizer().transform(["texto"])


def test_empty_fit_input_raises():
    with pytest.raises(ValueError):
        CountVectorizer().fit([])
    with pytest.raises(ValueError):
        TfidfVectorizer().fit([])
