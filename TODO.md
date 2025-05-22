### ✅ **TODO – Sprint 3: Correlação do Modelo com Dados Reais**

#### 📦 Repositório

* [x] Criar **novo repositório GitHub** exclusivo para a Sprint 3
* [x] Organizar por pastas: `data/`, `notebooks/`, `images/`, `scripts/`, etc.
* [ ] Documentar cada passo com README e comentários no código

---

#### 📥 Etapa 1 – Coleta de Dados Históricos

* [x] Pesquisar dados de produtividade, como em CONAB, IBGE, MAPA e CEPEA/USP
* [x] Filtrar por mesma **cultura** e **região** do modelo anterior (ex: milho em SP)
* [x] Selecionar variáveis:
  * [x] Produtividade média (t/ha)
  * [x] Ano/safra
  * [ ] Condições regionais (seca, chuva, etc)

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
* [ ] Fazer regressão linear simples:
  * [x] Gerar equação da tendência
  * [ ] Calcular R²
* [ ] Gerar gráficos:
  * [x] Dispersão com linha de tendência
  * [ ] Comparativos por safra ou região

---

#### 🧠 Etapa 4 – Interpretação Crítica

* [ ] Avaliar: NDVI foi um bom preditor?
* [ ] Identificar onde o modelo teve melhor/pior desempenho
* [ ] Discutir fatores externos (clima, pragas, imagem ruim)
* [ ] Sugerir melhorias no modelo:
  * [ ] Dados adicionais
  * [ ] Tratamento de imagem
  * [ ] Ajustes no período de coleta
* [ ] Discutir limitações:
  * [ ] Tamanho da amostra
  * [ ] Qualidade das fontes públicas
  * [ ] Métodos estatísticos usados

---

#### 📝 Entregáveis

* [ ] **Relatório técnico em PDF** com:
  * [ ] Metodologia de coleta
  * [ ] Técnicas estatísticas
  * [ ] Análise de gráficos
  * [ ] Discussão crítica
  * [ ] Referências dos dados
* [x] **Notebook Jupyter/Colab**:
  * [x] Scripts de tratamento
  * [x] Scripts de análise
  * [x] Geração de gráficos
  * [x] Link ou prints dos resultados

---

#### 👥 Organização Interna

* [x] Dividir funções: coleta / análise / interpretação
* [ ] Documentar tudo no GitHub com clareza
* [x] Evitar modificar repositórios de outras Sprints
* [ ] Registrar decisões e dificuldades enfrentadas

#### Pontos de melhoria:

- Faltou detalhamento na segmentação das áreas de cultivo — não foi mencionada a aplicação de máscara via NDVI ou qualquer abordagem de imageamento espacial. Mesmo que a abordagem tenha sido estatística, seria importante ao menos justificar essa ausência.
- Poderia haver uma visualização final mais robusta, como comparação entre os modelos lado a lado com gráficos de dispersão e linha temporal.
- O README poderia destacar com mais clareza os resultados numéricos finais das métricas (RMSE, MAE, R²) de forma tabulada e direta.
- A explicação do funcionamento do pipeline no notebook pode ser mais comentada em algumas partes, especialmente nas células que tratam dados climáticos.
