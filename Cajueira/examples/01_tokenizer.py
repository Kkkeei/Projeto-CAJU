from cajueira.nlp import normalize_text, tokenize

sentence = "Olá! Quero aprender Inteligência Artificial."
print("Original:", sentence)
print("Normalizado:", normalize_text(sentence))
print("Tokens:", tokenize(sentence))
