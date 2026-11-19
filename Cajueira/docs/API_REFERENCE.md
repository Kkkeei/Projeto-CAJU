# Referência inicial da API

Esta página descreve as APIs públicas de `cajueira` na versão `0.1.1`.

## 1. NLP

### `normalize_text(text, *, lowercase=True, strip_accents=False)`

Retorna um texto Unicode normalizado em NFC, com espaços consecutivos reduzidos a um único espaço. Por padrão, converte para minúsculas e preserva acentos.

### `tokenize(text, *, lowercase=True, strip_accents=False)`

Retorna uma lista de sequências alfanuméricas. Pontuação e `_` atuam como separadores. Esta implementação não reconhece entidades, faz stemming, lematização ou subword tokenization.

### `Vocabulary`

Construtor:

```python
Vocabulary(
    unk_token="<UNK>",
    pad_token=None,
    min_frequency=1,
    max_size=None,
)
```

Métodos e propriedades:

- `fit(texts)`: aprende o vocabulário a partir de textos.
- `fit_tokens(token_sequences)`: aprende de sequências já tokenizadas.
- `add_token(token)`: adiciona explicitamente um token.
- `encode(text, unknown="unk")`: converte texto em IDs; `unknown` aceita `unk`, `ignore` ou `error`.
- `decode(indices, skip_special_tokens=False)`: converte IDs de volta em tokens.
- `word_to_index`: mapa token → índice.
- `index_to_word`: lista índice → token.
- `save(path)` / `load(path)`: persiste o vocabulário em JSON.

A ordem padrão é a primeira ocorrência no corpus. `<UNK>` é reservado no índice zero por padrão. `<PAD>` é reservado antes de `<UNK>` quando configurado. `max_size` limita tokens comuns e não conta os tokens especiais.

**Importante:** o vocabulário não é um dicionário completo do português. Ele representa os tokens observados no corpus de treinamento. É necessário ajustar o vocabulário usando somente dados de treinamento e reutilizá-lo em validação, teste e inferência.

## 2. Vetorização

### `CountVectorizer(binary=False, min_frequency=1)`

- `fit(texts)`: constrói o vocabulário de atributos.
- `transform(texts)`: devolve uma matriz NumPy com contagens.
- `fit_transform(texts)`: ajusta e transforma.
- `get_feature_names_out()`: devolve os termos/colunas.

Com `binary=True`, cada posição indica presença (`1`) ou ausência (`0`), em vez da contagem.

### `TfidfVectorizer(sublinear_tf=False, norm="l2", min_frequency=1)`

Calcula frequência de termo e frequência inversa de documento suavizada:

\[
IDF(t) = \log\left(\frac{1 + N}{1 + DF(t)}\right) + 1
\]

onde `N` é o número de documentos e `DF(t)` é a quantidade de documentos que contêm o termo. O resultado padrão é normalizado pela norma L2 por documento. `norm=None` desativa a normalização. `sublinear_tf=True` usa `1 + log(tf)` para frequências positivas.

Os vetorizadores ignoram tokens que não estavam no corpus usado em `fit`. Eles não alteram o vocabulário durante `transform`.

## 3. Operações matemáticas

- `dot_product(a, b)`: produto escalar de vetores 1D de mesmo tamanho.
- `vector_norm(vector, p=2.0)`: norma Lp para `p >= 1`.
- `cosine_similarity(a, b)`: similaridade do cosseno. Retorna `0.0` quando pelo menos um vetor é nulo.

A similaridade de cosseno não é uma probabilidade e não mede significado semântico por si só.

## 4. Aprendizado clássico

### `MultinomialNaiveBayes(alpha=1.0)`

- `fit(X, y)`: treina em uma matriz 2D de atributos finitos e não negativos.
- `predict(X)`: retorna os rótulos estimados.
- `predict_proba(X)`: retorna probabilidades por classe.
- `predict_log_proba(X)`: retorna log-probabilidades normalizadas.

É um baseline útil para classificação de texto. `alpha` controla o suavizamento de Laplace/Lidstone e deve ser positivo.

## 5. Rede neural

### `MLPClassifier(...)`

Parâmetros principais:

- `hidden_sizes=(32,)`: número de neurônios em cada camada oculta.
- `learning_rate=0.01`: taxa de aprendizado.
- `epochs=100`: passagens completas pelo conjunto de treinamento.
- `batch_size=32`: tamanho máximo dos mini-lotes.
- `optimizer="adam"`: `adam` ou `sgd`.
- `random_state=42`: semente para reprodutibilidade.
- `l2=0.0`: coeficiente de regularização L2.

Métodos:

- `fit(X, y)`: treina a rede.
- `predict_proba(X)`: calcula a distribuição Softmax de saída.
- `predict(X)`: retorna os rótulos previstos.
- `score(X, y)`: calcula acurácia.

As camadas ocultas usam ReLU; a camada final usa Softmax; a perda é entropia cruzada. O código contém forward propagation, backpropagation e atualização por SGD ou Adam. A implementação é destinada a aprendizado e pequenos experimentos, não a treinamento de modelos grandes.

## 6. Roteamento

### `IntentRouter(threshold=0.15)`

- `add_intent(intent, examples)`: registra exemplos para uma intenção.
- `fit()`: calcula TF-IDF e centroides das intenções.
- `route(text)`: retorna `IntentResult` com `intent`, `similarity` e `scores`.

Quando a melhor similaridade fica abaixo do limiar, `intent` é `None`. O limiar precisa ser escolhido com dados de validação representativos. Como o método é lexical, ele pode falhar com sinônimos ou paráfrases que não compartilham palavras.
