# Relatório de auditoria técnica — Cajueira 0.1.0

**Data:** 19/11/2026  
**Escopo:** código-fonte e documentação contidos no `Caju.zip`, diretório `Caju/Cajueira-0.1.0/`.  
**Base GitHub:** `Kkkeei/Projeto-CAJU`, branch `main`, commit/tree consultado `74c695fc85ed7c25dbdf53d65da8b498ee756798`.

## 1. Resumo executivo

A Cajueira 0.1.0 é uma biblioteca Python pequena, modular e didática. A separação `nlp`, `vectors`, `math`, `learning`, `neural` e `routing` é adequada para a fundação atual. As dependências entre os módulos seguem uma direção coerente: vetorizadores usam NLP/vocabulário; o roteador compõe TF-IDF e matemática; os modelos recebem matrizes NumPy. Não foi identificado ciclo de importação no grafo de dependências examinado.

A execução inicial confirmou **44/44 testes existentes** e **6/6 exemplos**. Após a correção de falhas de validação e robustez, a suíte contém **62 testes aprovados**. O wheel Python também foi construído com sucesso.

**Conclusão:** a Cajueira é uma base funcional para experimentação clássica de NLP/ML em CPU. Não deve ser apresentada como sistema de IA conversacional, modelo generativo, framework completo de agentes ou solução de produção. A 0.1.x pode ser estabilizada como biblioteca de fundamentos, com as limitações descritas neste relatório.

## 2. Integridade e escopo da fonte

O ZIP anexado contém a implementação completa, documentação, testes, exemplos e arquivos auxiliares, mas também inclui ambiente virtual `.venv`, caches e metadados gerados. Esses artefatos não fazem parte da fonte que deve ser versionada.

A branch `main` consultada no GitHub contém somente `.gitignore`, `LICENSE` e `README.md`; ela **não contém o código-fonte da Cajueira 0.1.0**. Por isso, a auditoria foi baseada no ZIP anexado. A branch `main` não deve ser considerada equivalente ao ZIP.

## 3. Arquitetura e modularidade

### Pontos adequados

- `src/` separa o pacote instalável de testes, exemplos e documentação.
- `nlp` mantém normalização/tokenização/vocabulário separados da vetorização.
- `vectors` mantém Bag of Words e TF-IDF.
- `math` concentra operações vetoriais genéricas.
- `learning` e `neural` não dependem de interface, servidor ou roteamento.
- `routing` compõe abstrações existentes sem acoplar agente/UX.
- A API principal é reexportada por `cajueira.__init__`.

### Pontos para manter sob controle

- O pacote Python se chama `cajueira`, enquanto o nome de distribuição é `cajueira-nlp`. É válido, porém deve ser documentado para instalação/importação.
- `Vocabulary.fitted` é um atributo público mutável; chamadores podem alterar o estado da classe. A API ainda é pequena, então não foi mudado nesta rodada.
- Os vetorizadores geram matrizes densas `float64`. Isso é simples e apropriado aos exemplos, mas limita corpus e vocabulários grandes.
- O tokenizador divide por pontuação e não preserva URLs, e-mails, emojis, hashtags compostas, abreviações ou palavras hifenizadas como unidades especiais. Isso é uma limitação de escopo documentada, não um erro matemático.
- `CountVectorizer` e `TfidfVectorizer` têm APIs semelhantes, mas não possuem `get_feature_names_out` no formato de array NumPy; a lista Python foi mantida para preservar compatibilidade.

## 4. Auditoria matemática e algorítmica

### Normalização e tokenização

A normalização NFC, `lower()` opcional, remoção explícita de marcas diacríticas e colapso de espaços são coerentes. O comportamento padrão preserva acentos. A tokenização `[ ^\\W_ ]+` (sem os espaços desta representação) extrai sequências Unicode alfanuméricas e trata pontuação/underscore como separadores. É baseline lexical, não segmentador linguístico completo.

### Vocabulário

A contagem, frequência mínima, limite de tokens comuns, ordem de primeira ocorrência e índices especiais são coerentes. `<PAD>` e `<UNK>` são reservados antes dos tokens comuns. O JSON versionado evita desserializar estruturas arbitrárias de objetos Python. A validação original de `min_frequency`/`max_size` aceitava booleanos e floats; foi endurecida para inteiros e o `decode` passou a aceitar escalares inteiros NumPy.

### Bag of Words e TF-IDF

O TF-IDF usa:

`idf(t) = log((1 + N) / (1 + df(t))) + 1`

A frequência documental conta cada termo no máximo uma vez por documento; a frequência de termo conta ocorrências. A transformação sublinear é `1 + log(tf)` para tf positivo. A normalização L2 por documento trata linhas nulas sem divisão por zero. O ajuste do vocabulário/IDF acontece somente em `fit`, e `transform` ignora tokens não vistos, evitando adaptação inadvertida durante inferência. A fórmula e o tratamento são coerentes com a documentação.

### Operações vetoriais

Produto escalar, norma Lp e similaridade do cosseno verificam vetores 1D finitos e dimensões iguais. O cosseno define vetor nulo como similaridade zero. Foi identificado overflow na fórmula original para vetores finitos de magnitude extrema (`1e308`), capaz de produzir `NaN`; a implementação foi alterada para escalar os vetores individualmente antes da normalização e limitar o resultado à faixa matemática `[-1, 1]`. A norma também passou a rejeitar `p` não finito e a usar escalonamento para evitar overflow desnecessário.

### Multinomial Naive Bayes

A implementação calcula prior de classe, contagens por classe, suavização aditiva e log-probabilidade condicional por atributo. `predict_log_proba` usa log-sum-exp para normalizar as pontuações. A entrada de treino exige valores finitos e não negativos, compatíveis com Multinomial NB. A inferência também valida forma e valores. Foi endurecida a validação de `alpha`, que antes aceitava `NaN` e infinito, e definido retorno de matriz vazia com dimensões corretas ao prever lote vazio.

### MLP

A rede implementa camadas densas, ReLU nas camadas ocultas, Softmax numericamente estabilizado por subtração do máximo, entropia cruzada, derivada combinada Softmax + cross-entropy, backpropagation, SGD e Adam com correção de viés dos momentos. A regularização L2 é aplicada aos pesos, não aos biases, de forma consistente entre gradiente e valor da perda.

Foi realizada checagem de gradiente por diferenças finitas em todos os pesos e biases de uma rede pequena, incluindo L2. O erro absoluto máximo observado foi inferior a `1e-10`. Um teste automático de regressão equivalente foi incorporado.

Foram corrigidas validações de `hidden_sizes`, `epochs`, `batch_size`, `learning_rate` e `l2`: floats para dimensões/tamanhos, booleanos e valores não finitos não devem ser aceitos. `score` agora rejeita conjunto vazio em vez de retornar `NaN`.

Ainda não há persistência de pesos, early stopping, callbacks, validação durante o treinamento, normalização de atributos, GPU ou batching distribuído. A MLP é para dados pequenos/vetorizados.

### IntentRouter

O roteador ajusta um único TF-IDF nos exemplos cadastrados, calcula centroides por intenção e compara uma consulta usando cosseno. O limiar opera sobre similaridade lexical, não probabilidade. Foi encontrado um caso-limite: com threshold zero, mensagem vazia ou inteiramente OOV era atribuída à primeira intenção apesar de não haver evidência lexical. O roteamento agora rejeita essas mensagens.

## 5. Achados e correções desta rodada

| ID | Severidade | Achado | Tratamento |
|---|---|---|---|
| AUD-01 | Alta | Similaridade de cosseno podia retornar `NaN` em magnitudes extremas por overflow | Corrigido com escala independente e teste de valores grandes/pequenos |
| AUD-02 | Alta | `alpha` do Naive Bayes aceitava `NaN`/infinito | Corrigido com validação de finitude e positividade |
| AUD-03 | Alta | Hiperparâmetros da MLP aceitavam floats/bool em dimensões e valores não finitos | Corrigido com validação de tipos e faixas |
| AUD-04 | Média | Roteador podia escolher intenção para texto vazio/OOV com threshold zero | Corrigido e coberto por teste |
| AUD-05 | Média | Gradientes da MLP não tinham teste numérico de referência | Adicionado teste de diferenças finitas com L2 |
| AUD-06 | Média | `Vocabulary` aceitava limites não inteiros e `decode` rejeitava `numpy.int64` | Corrigido e coberto por testes |
| AUD-07 | Baixa | Naive Bayes tinha contrato de lote vazio pouco explícito | Retorno vazio com shape `(0, n_classes)` definido |
| AUD-08 | Baixa | `MLP.score` em lote vazio retornava `NaN` | Agora levanta `ValueError` |
| AUD-09 | Média | O ZIP trazia `LICENSE` com titular placeholder | Não levado à integração; a branch preserva a licença MIT preenchida que já existia na raiz da `main` (`Davi Lucas`); a titularidade legal continua sendo responsabilidade do mantenedor |
| AUD-10 | Média | ZIP inclui `.venv`, caches e metadados gerados | Excluir do versionamento; manter só fonte/testes/docs/exemplos |

## 6. Segurança e qualidade operacional

Não há execução de comandos externos, desserialização pickle, rede, servidor ou acesso a arquivos arbitrários durante as operações centrais. A serialização existente é JSON para vocabulário. Os principais riscos atuais são operacionais: recursos densos em memória, entrada de corpus sem limite de tamanho e falta de persistência segura/versionada dos modelos. Não foi feita análise de segurança dinâmica especializada; portanto, este relatório não constitui certificação de segurança.

## 7. Validação executada

Ambiente desta auditoria: Python 3.13, NumPy 2.3.5 e pytest 9.0.2.

- Baseline anexada: `44 passed`.
- Após correções e testes de regressão: `62 passed`.
- `compileall` em `src`, `tests` e `examples`: concluído.
- Seis exemplos: todos executaram com sucesso.
- Build de wheel com `pip wheel --no-deps --no-build-isolation`: concluído (`cajueira_nlp-0.1.1-py3-none-any.whl`).
- Checagem de gradientes com diferenças finitas: erro máximo observado abaixo de `1e-10`.

Não foram executados testes reais em Python 3.10, 3.11, 3.12 e 3.13 como matriz separada nesta sessão. O workflow declarado no repositório do ZIP cobre essas versões, mas execução local confirmada nesta auditoria foi somente Python 3.13. Também não houve medição formal de cobertura, benchmark, dataset externo ou avaliação de generalização.

## 8. Recomendações para 0.2.0

Priorizar, nesta ordem:

1. Métricas de classificação: accuracy, precision, recall, F1, matriz de confusão.
2. Estratégia de divisão treino/validação/teste e avaliação reproduzível.
3. Pipeline explícito para impedir vazamento de dados no `fit`.
4. Persistência versionada de vetorizadores, Naive Bayes e MLP (formato seguro, metadados e validação).
5. Baselines e experimentos com datasets representativos em português.
6. Vetorização esparsa quando volume justificar.
7. N-gramas e tokenizador configurável, sem substituir o baseline atual.
8. Calibração de threshold e avaliação de falsos positivos/negativos do IntentRouter.

Não adicionar Transformer, geração livre ou orquestração de agentes à Cajueira 0.1.x. Cajuína e Castanha devem consumir a biblioteca por interfaces explícitas em camadas próprias.

## 9. Resultado da rodada corretiva (candidata 0.1.1)

As correções e testes deste relatório foram organizados como uma atualização patch `0.1.1`, mantendo a auditoria da fonte original identificada como `0.1.0`. A versão Python do pacote e `cajueira.__version__` são sincronizadas em `0.1.1` na branch de evolução. Esta é uma candidata em branch, não uma publicação de pacote ou uma integração na `main`.

## 10. Critério de conclusão da 0.1.x

A fundação pode ser considerada estável para uso experimental quando: os testes automatizados passam em toda a matriz Python suportada; os exemplos e o wheel são reproduzíveis em ambiente limpo; documentação/API correspondem ao código; e as limitações de memória, escopo lexical e finalidade experimental são comunicadas. Os itens de CI na matriz ainda precisam de validação em GitHub Actions, e a titularidade legal da licença deve ser confirmada pelo mantenedor antes de uma publicação. A licença preenchida existente na raiz do repositório foi preservada na integração.
