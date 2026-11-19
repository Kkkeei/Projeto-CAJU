from cajueira.vectors import TfidfVectorizer
from cajueira.math import cosine_similarity

documents = [
    "gosto de inteligência artificial",
    "estudo programação em python",
    "inteligência artificial usa dados",
]

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(documents)
print("Termos:", vectorizer.get_feature_names_out())
print("Matriz TF-IDF:\n", matrix)
print("Similaridade entre documento 1 e 3:", cosine_similarity(matrix[0], matrix[2]))
