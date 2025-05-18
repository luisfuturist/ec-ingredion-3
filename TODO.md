### ✅ **TODO – Sprint 3: Correlação do Modelo com Dados Reais**

#### 📦 Repositório

* [x] Criar **novo repositório GitHub** exclusivo para a Sprint 3
* [x] Organizar por pastas: `data/`, `notebooks/`, `images/`, `scripts/`, etc.
* [ ] Documentar cada passo com README e comentários no código

---

#### 📥 Etapa 1 – Coleta de Dados Históricos

* [x] Pesquisar dados de produtividade, como em CONAB, IBGE, MAPA e CEPEA/USP
* [x] Filtrar por mesma **cultura** e **região** do modelo anterior (ex: milho em SP)
* [ ] Selecionar variáveis:
  * [x] Produtividade média (t/ha)
  * [x] Ano/safra
  * [ ] Condições regionais (seca, chuva, etc)

---

#### 🧹 Etapa 2 – Tratamento e Preparação dos Dados

* [ ] Construir tabela com:
  * [ ] Coluna de produtividade real
  * [ ] Coluna de NDVI médio correspondente
* [ ] Alinhar escalas temporais (ex: safra 2022/23)
* [ ] Tratar:
  * [ ] Dados ausentes
  * [ ] Outliers
  * [ ] Inconsistências
* [ ] Separar por amostras comparáveis (municípios, regiões, talhões)

---

#### 📊 Etapa 3 – Análise Estatística

* [ ] Calcular correlação:
  * [ ] Pearson (opcional)
  * [ ] Spearman (opcional)
* [ ] Interpretar força da correlação (forte, moderada, fraca)
* [ ] Fazer regressão linear simples:
  * [ ] Gerar equação da tendência
  * [ ] Calcular R²
* [ ] Gerar gráficos:
  * [ ] Dispersão com linha de tendência
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
* [ ] **Notebook Jupyter/Colab**:
  * [ ] Scripts de tratamento
  * [ ] Scripts de análise
  * [ ] Geração de gráficos
  * [ ] Link ou prints dos resultados

---

#### 👥 Organização Interna

* [ ] Dividir funções: coleta / análise / interpretação
* [ ] Documentar tudo no GitHub com clareza
* [ ] Evitar modificar repositórios de outras Sprints
* [ ] Registrar decisões e dificuldades enfrentadas

#### Pontos de melhoria:

- Faltou detalhamento na segmentação das áreas de cultivo — não foi mencionada a aplicação de máscara via NDVI ou qualquer abordagem de imageamento espacial. Mesmo que a abordagem tenha sido estatística, seria importante ao menos justificar essa ausência.
- Poderia haver uma visualização final mais robusta, como comparação entre os modelos lado a lado com gráficos de dispersão e linha temporal.
- O README poderia destacar com mais clareza os resultados numéricos finais das métricas (RMSE, MAE, R²) de forma tabulada e direta.
- A explicação do funcionamento do pipeline no notebook pode ser mais comentada em algumas partes, especialmente nas células que tratam dados climáticos.
