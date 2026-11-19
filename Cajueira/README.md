# Cajueira 🌳

**Biblioteca Python modular de processamento de linguagem natural (NLP), vetorização e aprendizado de máquina clássico.**

Cajueira é o motor computacional do ecossistema CAJU. Esta primeira versão reúne componentes clássicos implementados de forma explícita, com NumPy, para que possam ser estudados, testados e evoluídos.

> Estado: `0.1.1` (fundação experimental; revisão corretiva sobre a base 0.1.0). A auditoria desta entrega encontrou e corrigiu problemas de validação de entradas e estabilidade numérica; os detalhes estão em `docs/AUDIT_0.1.0.md`. A biblioteca ainda não é um modelo generativo nem um sistema de IA de produção.

## O que já existe

- Normalização Unicode e tokenização de texto.
- Vocabulário treinável com `<UNK>`, `<PAD>`, frequência mínima, limite de tamanho e serialização JSON.
- Bag of Words e TF-IDF.
- Produto escalar, norma vetorial e similaridade de cosseno.
- Classificador Multinomial Naive Bayes.
- MLP multiclasses com NumPy, ReLU, Softmax, backpropagation, SGD ou Adam.
- Roteador de intenções usando centroides TF-IDF.
- Testes automatizados, exemplos e documentação da arquitetura.
- API NumPy com validação explícita de entradas e operações vetoriais estáveis para magnitudes grandes e pequenas.

## O que ainda não faz

- Não gera respostas em linguagem natural de forma aberta.
- Não possui embeddings semânticos ou um Transformer.
- Não tem memória conversacional, chamadas de ferramentas ou orquestração de agentes.
- O tokenizador não faz lematização, stemming ou segmentação em subpalavras.

Esses limites são intencionais: os componentes atuais são baselines para medir e compreender antes de adicionar modelos mais complexos.

## Requisitos

- Python 3.10 ou superior.
- NumPy 1.24 ou superior.

## Instalação para desenvolvimento

```bash
git clone https://github.com/Kkkeei/Projeto-CAJU.git
cd Projeto-CAJU/Cajueira
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Uso rápido

```python
from cajueira import Vocabulary, TfidfVectorizer, cosine_similarity

texts = [
    "eu estudo inteligência artificial",
    "eu programo em python",
    "inteligência artificial usa dados",
]

vocabulary = Vocabulary().fit(texts)
print(vocabulary.word_to_index)
print(vocabulary.encode("eu estudo python"))

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(texts)
print(vectorizer.get_feature_names_out())
print(cosine_similarity(matrix[0], matrix[2]))
```

## Exemplo: roteamento de intenções

```python
from cajueira import IntentRouter

router = IntentRouter(threshold=0.12)
router.add_intent("saudacao", ["olá", "oi", "bom dia"])
router.add_intent("ajuda", ["preciso de ajuda", "tenho uma dúvida"])
router.add_intent("despedida", ["tchau", "até logo"])
router.fit()

result = router.route("oi, bom dia")
print(result.intent)
print(result.similarity)  # similaridade lexical, não probabilidade
```

## Estrutura do projeto

```text
Cajueira/
├── src/cajueira/
│   ├── nlp/          # normalização, tokenização, vocabulário
│   ├── vectors/      # Bag of Words e TF-IDF
│   ├── math/         # operações vetoriais
│   ├── learning/     # algoritmos clássicos de ML
│   ├── neural/       # rede MLP com NumPy
│   └── routing/      # roteamento de intenções
├── tests/            # testes automatizados
├── examples/         # exemplos executáveis
├── docs/             # arquitetura, glossário e desenvolvimento
├── pyproject.toml
└── CHANGELOG.md
```

## Testes

```bash
pytest -v
```

## Estado e limites da versão 0.1.x

A Cajueira 0.1.x entrega baselines clássicos pequenos, executados em CPU com NumPy. As matrizes dos vetorizadores são densas; portanto, o custo de memória cresce com `n_documentos × n_atributos`. A MLP não implementa persistência de pesos, minibatches distribuídos, GPU, early stopping ou callbacks. O roteador retorna uma pontuação lexical de similaridade, não uma probabilidade calibrada. Os testes verificam contratos e propriedades numéricas, mas não demonstram generalização em dados reais: isso exige datasets representativos e avaliação separada.

## Documentação adicional

- [Arquitetura](docs/ARCHITECTURE.md)
- [Relatório de auditoria 0.1.0](docs/AUDIT_0.1.0.md)
- [Glossário](docs/GLOSSARY.md)
- [Guia de desenvolvimento](docs/DEVELOPMENT.md)
- [Referência da API](docs/API_REFERENCE.md)
- [Roteiro de revisão técnica](docs/REVIEW_GUIDE.md)
- [Histórico de mudanças](CHANGELOG.md)

## Roadmap

- [x] Estrutura de pacote Python (`src/`).
- [x] Normalização e tokenização Unicode.
- [x] Vocabulário treinável e serializável.
- [x] Bag of Words e TF-IDF.
- [x] Operações vetoriais.
- [x] Baseline Multinomial Naive Bayes.
- [x] MLP educacional em NumPy.
- [x] Roteador de intenções lexical.
- [ ] Métricas de classificação e avaliação treino/validação/teste.
- [ ] Persistência de vetorizadores e modelos.
- [ ] Pipeline de pré-processamento e treinamento.
- [ ] Embeddings e experimentos semânticos.
- [ ] Integração com ferramentas e agentes do ecossistema CAJU.

## Licença

A licença do componente segue a licença MIT declarada na raiz do repositório CAJU; consulte `../LICENSE`.
