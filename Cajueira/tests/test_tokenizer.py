import pytest

from cajueira.nlp.tokenizer import normalize_text, tokenize


def test_normalize_text_lowercases_and_collapses_spaces():
    assert normalize_text("  OLA   MUNDO!  ") == "ola mundo!"


def test_unicode_nfc_normalization():
    decomposed = "cafe\u0301"
    assert normalize_text(decomposed) == "café"


def test_preserves_accents_by_default():
    assert tokenize("avó e avô") == ["avó", "e", "avô"]


def test_can_strip_accents_explicitly():
    assert tokenize("ação", strip_accents=True) == ["acao"]


def test_tokenize_words_numbers_and_punctuation():
    assert tokenize("Quero programar em Python! Tenho 3 projetos.") == [
        "quero", "programar", "em", "python", "tenho", "3", "projetos"
    ]


def test_underscore_is_separator():
    assert tokenize("caju_ina") == ["caju", "ina"]


def test_empty_text():
    assert tokenize("") == []


def test_invalid_input():
    with pytest.raises(TypeError):
        tokenize(123)
