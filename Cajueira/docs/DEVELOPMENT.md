# Guia de desenvolvimento

## Ambiente suportado

- Python 3.10 ou superior.
- NumPy 1.24 ou superior.
- Pytest 8 ou superior para desenvolvimento.

Crie um ambiente isolado e instale o pacote em modo editável:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Testes e verificações

```bash
pytest -v
python -m compileall -q src tests examples
```

Antes de uma versão candidata, execute os seis exemplos:

```bash
for example in examples/0*.py; do
  python "$example"
done
```

Para construir e validar o pacote de distribuição:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

## Princípios para contribuições

- Toda funcionalidade nova deve incluir testes de contrato, limites e resultados matemáticos.
- Funções e classes públicas devem documentar entradas, saídas, exceções e limitações.
- Não ajuste vocabulário/IDF com dados de validação ou teste.
- Use geradores aleatórios locais e exponha `random_state` quando houver aleatoriedade.
- Mantenha o pacote central independente de interface, servidor, conectores e agentes.
- Não use `pickle` para carregar artefatos não confiáveis.
- Atualize referência da API, README e changelog quando alterar comportamento público.
- Evite adicionar dependências sem demonstrar necessidade.
