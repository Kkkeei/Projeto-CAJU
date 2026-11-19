from cajueira.nlp import Vocabulary

texts = [
    "eu gosto de python",
    "eu estudo inteligência artificial",
    "eu gosto de inteligência artificial",
]

vocabulary = Vocabulary(pad_token="<PAD>").fit(texts)
print("Vocabulário:", vocabulary.word_to_index)
print("Tamanho:", len(vocabulary))

encoded = vocabulary.encode("eu gosto de python")
print("Codificado:", encoded)
print("Decodificado:", vocabulary.decode(encoded))

vocabulary.save("vocabulary.json")
