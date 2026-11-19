import pytest

from cajueira.nlp.vocabulary import Vocabulary


def test_fit_preserves_first_seen_order_and_removes_duplicates():
    vocabulary = Vocabulary().fit(["eu gosto de python", "eu gosto de IA"])
    assert vocabulary.index_to_word == [
        "<UNK>", "eu", "gosto", "de", "python", "ia"
    ]
    assert vocabulary["gosto"] == 2


def test_encode_and_decode():
    vocabulary = Vocabulary().fit(["casa carro computador"])
    encoded = vocabulary.encode("casa computador")
    assert vocabulary.decode(encoded) == ["casa", "computador"]


def test_unknown_token_uses_unk_by_default():
    vocabulary = Vocabulary().fit(["casa carro"])
    assert vocabulary.encode("casa avião") == [1, 0]


def test_unknown_token_can_be_ignored_or_raise():
    vocabulary = Vocabulary().fit(["casa carro"])
    assert vocabulary.encode("casa avião", unknown="ignore") == [1]
    with pytest.raises(KeyError):
        vocabulary.encode("avião", unknown="error")


def test_unknown_policy_is_validated():
    vocabulary = Vocabulary().fit(["casa"])
    with pytest.raises(ValueError):
        vocabulary.encode("casa", unknown="skip")


def test_min_frequency_and_max_size():
    vocabulary = Vocabulary(min_frequency=2, max_size=1).fit(
        ["gato gato cão", "gato pássaro"]
    )
    assert vocabulary.index_to_word == ["<UNK>", "gato"]


def test_no_unk_token_means_unknowns_are_ignored_by_default():
    vocabulary = Vocabulary(unk_token=None).fit(["casa"])
    assert vocabulary.encode("casa avião") == [0]


def test_special_tokens_are_reserved():
    vocabulary = Vocabulary(unk_token="<UNK>", pad_token="<PAD>").fit(
        ["<unk> gato"]
    )
    assert vocabulary.index_to_word[:2] == ["<PAD>", "<UNK>"]


def test_serialization_round_trip(tmp_path):
    original = Vocabulary(pad_token="<PAD>").fit(["olá mundo olá"])
    path = tmp_path / "vocabulary.json"
    original.save(path)
    restored = Vocabulary.load(path)
    assert restored.to_dict() == original.to_dict()
    assert restored.encode("mundo desconhecido") == original.encode("mundo desconhecido")


def test_invalid_configuration():
    with pytest.raises(ValueError):
        Vocabulary(min_frequency=0)
    with pytest.raises(ValueError):
        Vocabulary(max_size=0)


def test_invalid_token_and_index():
    vocabulary = Vocabulary().fit(["casa"])
    with pytest.raises(ValueError):
        vocabulary.add_token("")
    with pytest.raises(IndexError):
        vocabulary.decode([999])


def test_vocabulary_rejects_non_integer_limits():
    with pytest.raises(TypeError):
        Vocabulary(min_frequency=1.5)
    with pytest.raises(TypeError):
        Vocabulary(max_size=2.5)


def test_decode_accepts_numpy_integer_scalars():
    import numpy as np

    vocabulary = Vocabulary().fit(["casa"])
    assert vocabulary.decode([np.int64(0)]) == ["<UNK>"]
