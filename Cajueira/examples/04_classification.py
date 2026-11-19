from cajueira.learning import MultinomialNaiveBayes
from cajueira.vectors import TfidfVectorizer

texts = [
    "quero comprar um carro",
    "meu veículo precisa de manutenção",
    "quero uma receita de bolo",
    "como preparar uma comida",
    "comprar automóvel usado",
    "receita de pão caseiro",
]
labels = ["veiculos", "veiculos", "culinaria", "culinaria", "veiculos", "culinaria"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
classifier = MultinomialNaiveBayes().fit(X, labels)

query = ["preciso consertar meu carro"]
prediction = classifier.predict(vectorizer.transform(query))
print("Intenção prevista:", prediction[0])
