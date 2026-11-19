# Arquitetura da Cajueira 0.1.1

## Objetivo

Cajueira é uma biblioteca Python para estudar e compor componentes clássicos de processamento de linguagem natural (NLP), vetorização e aprendizado de máquina. A biblioteca é o motor computacional; uma interface conversacional ou um agente de programação são aplicações separadas que podem consumir suas APIs.

## Fluxo básico de texto

```text
Texto bruto
   |
   v
normalize_text / tokenize
   |
   v
Vocabulary ----------------------> IDs de tokens
   |
   +--> CountVectorizer ----------> matriz Bag of Words
   |
   +--> TfidfVectorizer ----------> matriz TF-IDF
                                      |
                         +------------+-------------+
                         |                          |
                         v                          v
                  Naive Bayes                   MLPClassifier
                         |
                         +------------+-------------+
                                      v
                                previsão de classe

TfidfVectorizer --> IntentRouter --> intenção mais próxima / sem correspondência
```

## Módulos

- `nlp`: normalização, tokenização e vocabulário treinável.
- `vectors`: Bag of Words e TF-IDF.
- `math`: produto escalar, norma e similaridade de cosseno.
- `learning`: classificador Multinomial Naive Bayes.
- `neural`: MLP multiclasses implementada com NumPy.
- `routing`: roteador de intenções baseado em centroides TF-IDF.

## Decisões de projeto

1. **`src/` layout:** separa pacote instalável de testes e documentação.
2. **NumPy:** usado para matrizes e operações vetorizadas; não há dependência de frameworks de deep learning.
3. **Vocabulário ajustado aos dados:** o vocabulário é aprendido no conjunto de treinamento, não é uma lista universal do idioma.
4. **Tokens especiais:** `<UNK>` representa palavras fora do vocabulário; `<PAD>` pode ser reservado para operações futuras com sequências de tamanho fixo.
5. **Semântica lexical explícita:** TF-IDF e cosseno detectam sobreposição lexical, não compreensão semântica profunda.
6. **Separação de camadas:** o roteador e os classificadores não dependem de interface, servidor HTTP ou produto conversacional.

## Limitações conhecidas

- O tokenizador é baseado em expressões regulares e não faz lematização, stemming, análise morfossintática ou segmentação em subpalavras.
- TF-IDF não entende sinônimos sem sobreposição de termos.
- A MLP é uma implementação pequena para fins educacionais; não tem GPU, checkpoint avançado, regularização sofisticada, batching otimizado ou suporte a treinamento distribuído.
- O roteador usa similaridade de cosseno com centroides e não produz probabilidades calibradas.
- Não existe geração livre de linguagem natural nesta versão.

## Próximas linhas de evolução

1. Pipelines e transformação de datasets sem vazamento entre treino e teste.
2. Métricas: precisão, recall, F1, matriz de confusão e validação cruzada.
3. Persistência versionada de vetorizadores e modelos.
4. Stopwords e pré-processamento configurável por idioma.
5. Subword tokenization e embeddings.
6. Experimentos reprodutíveis e comparação de baselines.
7. Interfaces de agentes e ferramentas em um pacote separado.
