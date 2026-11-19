import pytest

from cajueira.math import cosine_similarity, dot_product, vector_norm


def test_dot_product():
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32.0


def test_vector_norm():
    assert vector_norm([3, 4]) == 5.0


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1, 2], [1, 2]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero():
    assert cosine_similarity([0, 0], [1, 2]) == 0.0


def test_incompatible_vectors_raise():
    with pytest.raises(ValueError):
        dot_product([1], [1, 2])


def test_cosine_similarity_is_stable_for_very_large_vectors():
    assert cosine_similarity([1e308, 1e308], [1e308, 1e308]) == pytest.approx(1.0)


def test_cosine_similarity_is_stable_for_very_small_vectors():
    assert cosine_similarity([1e-300, 0.0], [-1e-300, 0.0]) == pytest.approx(-1.0)


def test_vector_norm_rejects_non_finite_order():
    with pytest.raises(ValueError):
        vector_norm([1.0, 2.0], p=float("nan"))
