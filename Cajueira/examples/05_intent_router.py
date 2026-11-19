from cajueira.routing import IntentRouter

router = IntentRouter(threshold=0.12)
router.add_intent("saudacao", ["olá", "oi", "bom dia", "boa tarde"])
router.add_intent("ajuda", ["preciso de ajuda", "pode me ajudar", "tenho uma dúvida"])
router.add_intent("despedida", ["tchau", "até logo", "vou embora"])
router.fit()

for message in ["oi, tudo bem?", "pode me ajudar com uma dúvida?", "até amanhã"]:
    result = router.route(message)
    print(message, "->", result.intent, f"(similaridade={result.similarity:.3f})")
