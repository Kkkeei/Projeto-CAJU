# Roteiro de revisão técnica da Cajueira

Use este documento para estudar a implementação e decidir como melhorá-la. Não trate a versão 0.1.1 como arquitetura definitiva.

## 1. Tokenização e vocabulário

- A expressão regular atual é adequada para URLs, e-mails, abreviações, emojis, hashtags e palavras com hífen?
- Separar pontuação sempre é desejável? Em quais tarefas `!!!`, `?` ou emojis carregam intenção?
- Quando remover acentos ajuda e quando destrói informação?
- O token especial `<UNK>` deve ser um único índice ou seria melhor usar subpalavras?
- Como o vocabulário se comporta com corpus muito grande, `min_frequency` e `max_size`?
- O formato JSON do vocabulário precisa armazenar também a configuração do tokenizador?

## 2. Vetorização e vazamento de dados

- O vocabulário e o IDF devem ser ajustados somente no conjunto de treinamento?
- O que muda se `fit` for chamado com treino + teste?
- Bag of Words e TF-IDF perdem a ordem das palavras. Quais erros isso pode causar?
- Qual o custo de memória de uma matriz densa? Quando seria necessário usar matrizes esparsas?
- Como comparar TF-IDF com n-gramas de palavras ou caracteres?

## 3. Modelos de classificação

- Estabeleça baselines antes de usar uma rede neural: maioria de classes, Naive Bayes e depois MLP.
- Separe treino, validação e teste. Não use o teste para escolher hiperparâmetros.
- A acurácia é suficiente quando as classes são desbalanceadas?
- Adicione precisão, recall, F1 por classe e matriz de confusão.
- Compare o modelo com uma baseline simples e registre os resultados.

## 4. MLP e matemática

- Verifique as dimensões de cada matriz de pesos e vetor de bias.
- Faça derivação manual da entropia cruzada + Softmax e da derivada de ReLU.
- Compare gradientes calculados por backpropagation com diferenças finitas num teste numérico.
- Observe a curva de perda e identifique divergência, estagnação e overfitting.
- Teste diferentes inicializações, taxas de aprendizado, tamanhos de batch e regularização.
- Introduza normalização de atributos quando ela fizer sentido para os dados.

## 5. Roteador de intenções

- Crie exemplos variados para cada intenção, incluindo formas curtas, longas e erros de digitação.
- Calibre `threshold` em um conjunto de validação, não por tentativa apenas no conjunto de treino.
- Meça falsos positivos e falsos negativos.
- Defina como tratar intenções ambíguas ou múltiplas intenções.
- Compare centroides com classificador supervisionado.

## 6. Engenharia de software

- A API é consistente entre `fit`, `transform`, `predict` e `save/load`?
- Os erros são claros e previsíveis?
- Os modelos treinados podem ser persistidos de forma segura e versionada?
- Os testes cobrem limites, entradas vazias, dados inválidos e reprodutibilidade?
- O README descreve o que o código realmente faz?
- Mudanças públicas exigem atualização de versão e changelog?

## 7. Próxima entrega recomendada

Antes de adicionar Transformers ou geração de texto, uma boa próxima versão seria implementar avaliação de modelos e um pipeline reprodutível:

1. divisão treino/validação/teste;
2. `fit` de pré-processadores somente no treino;
3. transformações consistentes para validação e teste;
4. métricas por classe;
5. comparação de baselines;
6. registro dos experimentos e configurações.
