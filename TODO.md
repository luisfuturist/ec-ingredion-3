### ✅ **TODO – Sprint 3: Correlação do Modelo com Dados Reais**

#### 📦 Repositório

* [x] Criar **novo repositório GitHub** exclusivo para a Sprint 3
* [x] Organizar por pastas: `data/`, `notebooks/`, `images/`, `scripts/`, etc.

---

#### 📥 Etapa 1 – Coleta de Dados Históricos

* [x] Pesquisar dados de produtividade, como em CONAB, IBGE, MAPA e CEPEA/USP
* [x] Filtrar por mesma **cultura** e **região** do modelo anterior (ex: milho em SP)
* [x] Selecionar variáveis:
  * [x] Produtividade média (t/ha)
  * [x] Ano/safra

---

#### 🧹 Etapa 2 – Tratamento e Preparação dos Dados

* [x] Construir tabela com:
  * [x] Coluna de produtividade real
  * [x] Coluna de NDVI médio correspondente
* [x] Alinhar escalas temporais (ex: safra 2022/23)
* [x] Tratar:
  * [x] Dados ausentes
  * [x] Outliers
* [x] Separar por amostras comparáveis (municípios, regiões, talhões)

---

#### 📊 Etapa 3 – Análise Estatística

* [x] Calcular correlação, eg. Pearson, Spearman
* [x] Interpretar força da correlação (forte, moderada, fraca)
* [x] Fazer regressão linear simples:
  * [x] Gerar equação da tendência
  * [x] Calcular R²
* [x] Gerar gráficos:
  * [x] Dispersão com linha de tendência
  * (Não se aplica) [-] Comparativos por safra ou região

---

#### 🧠 Etapa 4 – Interpretação Crítica

* [x] Avaliar: NDVI foi um bom preditor?
* [x] Identificar onde o modelo teve melhor/pior desempenho
* [x] Discutir fatores externos (clima, pragas, imagem ruim)
* [x] Sugerir melhorias no modelo:
  * [x] Dados adicionais
  * [x] Tratamento de imagem
  * [x] Ajustes no período de coleta
* [x] Discutir limitações:
  * [x] Tamanho da amostra
  * [x] Qualidade das fontes públicas
  * [x] Métodos estatísticos usados

---

#### 📝 Entregáveis

* [x] **Relatório técnico em PDF** com:
  * [x] Metodologia de coleta
  * [x] Técnicas estatísticas
  * [x] Análise de gráficos
  * [x] Discussão crítica
  * [x] Referências dos dados
* [x] **Notebook Jupyter/Colab**:
  * [x] Scripts de tratamento
  * [x] Scripts de análise
  * [x] Geração de gráficos
  * [x] Link ou prints dos resultados

---

#### 👥 Organização Interna

* [x] Dividir funções: coleta / análise / interpretação
* [x] Evitar modificar repositórios de outras Sprints
* [x] Documentar tudo no GitHub com clareza
* [x] Registrar decisões e dificuldades enfrentadas

#### Pontos de melhoria:

- [x] Faltou detalhamento na segmentação das áreas de cultivo — não foi mencionada a aplicação de máscara via NDVI ou qualquer abordagem de imageamento espacial. Mesmo que a abordagem tenha sido estatística, seria importante ao menos justificar essa ausência.
- [ ] Poderia haver uma visualização final mais robusta, como comparação entre os modelos lado a lado com gráficos de dispersão e linha temporal.
- [ ] A documentação poderia destacar com mais clareza os resultados numéricos finais das métricas (RMSE, MAE, R²) de forma tabulada e direta.

- (INVALID) [-] A explicação do funcionamento do pipeline no notebook pode ser mais comentada em algumas partes, especialmente nas células que tratam dados climáticos.
