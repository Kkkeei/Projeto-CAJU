import numpy as np

from cajueira.neural import MLPClassifier

# Pequeno conjunto sintético: duas classes separáveis.
X = np.array([
    [-2.0, -1.0], [-1.5, -2.0], [-1.0, -1.0],
    [1.0, 1.0], [1.5, 2.0], [2.0, 1.0],
])
y = np.array(["grupo_a", "grupo_a", "grupo_a", "grupo_b", "grupo_b", "grupo_b"])

model = MLPClassifier(
    hidden_sizes=(8,),
    learning_rate=0.03,
    epochs=200,
    batch_size=6,
    random_state=7,
).fit(X, y)

print("Acurácia de treinamento:", model.score(X, y))
print("Previsões:", model.predict(X))
print("Última perda:", model.loss_history_[-1])
