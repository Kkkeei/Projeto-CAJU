# Changelog

Todas as mudanças importantes da Cajueira serão registradas aqui.

## [0.1.1] - 2026-11-19

### Corrigido
- Estabilizada a similaridade de cosseno para vetores com magnitudes muito grandes ou pequenas.
- Validada a finitude e o tipo dos hiperparâmetros do Naive Bayes e da MLP.
- Corrigida a aceitação de mensagens vazias/OOV pelo roteador com threshold zero.
- Endurecida a validação de limites do vocabulário e ampliado o suporte a inteiros NumPy no decode.
- Definido o contrato de predição de lote vazio no Naive Bayes e rejeitado score de MLP em conjunto vazio.

### Testes
- Ampliada a suíte de 44 para 62 testes.
- Adicionada verificação de gradientes da MLP por diferenças finitas, incluindo regularização L2.
- Reexecutados os seis exemplos e validado o build do wheel.

### Documentação
- Adicionado relatório de auditoria técnica.
- Documentados escopo, limitações e critérios de estabilização.

## [0.1.0] - 2026-11-19

### Adicionado
- Normalização Unicode e tokenização básica.
- Vocabulário treinável com tokens especiais, frequência mínima e limite de tamanho.
- Vetorizadores Bag of Words e TF-IDF.
- Operações vetoriais e similaridade de cosseno.
- Classificador Multinomial Naive Bayes.
- Classificador MLP multiclasses com NumPy e otimizador SGD/Adam.
- Roteador de intenções baseado em centroides TF-IDF.
- Testes automatizados, exemplos e documentação inicial.

### Observação
Esta é uma versão inicial educacional. Os componentes são modelos clássicos e ferramentas de base; não constituem um modelo generativo de linguagem de grande escala.
