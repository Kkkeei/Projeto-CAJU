# Projeto-CAJU

**Células Artificiais de Junção Unificada** — ecossistema modular de inteligência artificial em Python/NumPy.

O CAJU organiza componentes com responsabilidades específicas:

- **Cajueira:** motor computacional e biblioteca central de NLP, matemática, aprendizado de máquina e infraestrutura de modelos clássicos.
- **Cajuína:** agente conversacional, responsável pela experiência de diálogo, contexto, memória e uso de ferramentas.
- **Castanha:** agente especializado em programação, responsável por analisar, escrever, testar e corrigir código.

## Arquitetura

A visão inicial do ecossistema separa três funções:

1. **Motor e roteamento:** transforma entradas em representações processáveis e encaminha tarefas.
2. **Agente conversacional:** conversa com o usuário e coordena contexto e ferramentas.
3. **Agente de programação:** executa fluxos especializados de desenvolvimento de software.

A Cajueira fornece a fundação algorítmica, mas não substitui os agentes Cajuína e Castanha. Interfaces conversacionais, memória de diálogo, execução de ferramentas e orquestração de agentes permanecem em camadas próprias.

## Componente disponível

### Cajueira

A primeira fundação implementada está em [`Cajueira/`](Cajueira/).

- [README e instalação](Cajueira/README.md)
- [Arquitetura](Cajueira/docs/ARCHITECTURE.md)
- [Relatório de auditoria técnica](Cajueira/docs/AUDIT_0.1.0.md)
- [Referência da API](Cajueira/docs/API_REFERENCE.md)
- [Histórico de versões](Cajueira/CHANGELOG.md)

A Cajueira 0.1.x inclui tokenização Unicode básica, vocabulário treinável, Bag of Words, TF-IDF, operações vetoriais, Multinomial Naive Bayes, MLP com NumPy e roteamento lexical de intenções. Ela não é um LLM, não gera linguagem livre e não oferece ainda runtime de agentes.

## Princípios do repositório

- Manter os componentes desacoplados por APIs explícitas.
- Evitar dependências pesadas sem necessidade demonstrada.
- Documentar limitações e decisões arquiteturais.
- Não confundir exemplos funcionais com evidência de generalização.
- Preservar histórico e evoluir componentes em branches e pull requests.

## Licença

MIT. Consulte [`LICENSE`](LICENSE).
