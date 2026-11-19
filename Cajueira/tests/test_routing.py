import pytest

from cajueira.routing import IntentRouter


def test_intent_router_routes_matching_examples():
    router = IntentRouter(threshold=0.05)
    router.add_intent("saudacao", ["olá", "bom dia", "oi tudo bem"])
    router.add_intent("despedida", ["tchau", "até logo", "até amanhã"])
    router.fit()
    result = router.route("olá, bom dia")
    assert result.intent == "saudacao"
    assert result.similarity > 0
    assert set(result.scores) == {"saudacao", "despedida"}


def test_intent_router_rejects_unmatched_text():
    router = IntentRouter(threshold=0.9)
    router.add_intent("saudacao", ["olá", "bom dia"])
    router.add_intent("despedida", ["tchau", "até logo"])
    router.fit()
    assert router.route("mecânica quântica").intent is None


def test_router_must_be_fitted():
    router = IntentRouter().add_intent("saudacao", ["oi"])
    with pytest.raises(RuntimeError):
        router.route("oi")


def test_router_requires_examples():
    with pytest.raises(ValueError):
        IntentRouter().fit()


def test_empty_or_oov_text_is_not_routed_even_with_zero_threshold():
    router = IntentRouter(threshold=0.0)
    router.add_intent("saudacao", ["oi", "olá"]).fit()
    assert router.route("").intent is None
    assert router.route("termo completamente desconhecido").intent is None
