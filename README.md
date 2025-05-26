# ec-ingredion-3

> Este projeto faz parte do curso de **Inteligência Artificial** oferecido pela [FIAP](https://github.com/fiap) - Online 2024. Este repositório reúne os materiais relacionados ao **Enterprise Challenge - Ingredion**, correspondendo à **Sprint 3** do desafio.

- Notebook para Machine Learning: [Jupyter Notebook](./notebooks/ml.ipynb)
- Notebook de Análise Exploratória de Dados: [Jupyter Notebook](./notebooks/eda.ipynb)
- 📄 **Relatório Final:** [Relatório Técnico - Sprint 3](inserir_link_relatorio)

## Descrição

Este projeto dá continuidade ao desenvolvimento de um modelo de Inteligência Artificial para prever a produtividade agrícola, focando na cultura do café na região de Manhuaçu (MG). Na Sprint 3, o objetivo foi validar o modelo com dados reais históricos e analisar o comportamento das previsões, substituindo abordagens tradicionais por uma solução baseada em NDVI e dados históricos.

A solução proposta visa:

* **Otimizar o planejamento agrícola:** Fornecendo previsões de produtividade mais acuradas, auxiliando na alocação eficiente de recursos (fertilizantes, mão de obra, etc.).
* **Reduzir perdas:** Identificando tendências e possíveis impactos de fatores ambientais (seca, pragas, etc.) na produção.
* **Adaptabilidade e escalabilidade:** A solução é projetada pode ser adaptada a diferentes culturas e escalável para outras regiões, aumentando seu valor estratégico.

## Estrutura de Arquivos

```py
├── data                  # Arquivos de entrada e saída usados no processo
│   ├── PROCESSED           # Dados pré-processados para os modelos
│   │   ├── manhuacu.csv  # Produção histórica (1974-2023) + NDVI anual médio
│   │   └── ndvi.csv      # Série temporal NDVI (2000-2023) com colunas cíclicas
│   ├── GOOGLE_EARTH_ENGINE # Dados NDVI extraídos do Google Earth Engine
│   │   └── ndvi_manhuacu.csv # Série NDVI (2000–2025): Google Earth Engine
│   └── SIDRA               # Dados de produção do IBGE
│       └── tabela1613.xlsx # Dados de produção (1974–2023): IBGE/Tabela 1613
├── models                # Arquivos de pesos dos modelos treinados
│   ├── lstm.pth        # Pesos do modelo LSTM
│   └── mlp.pth         # Pesos do modelo MLP
├── README.md             # Este README
├── requirements.txt      # Lista de dependências do projeto
├── scripts               # Notebooks para extração e preparação dos dados
│   └── extract-analysis-data.ipynb           # Preparação e integração dos dados para análise
│   └── extract-ndvi-manhuacu.ipynb # Extração NDVI de Manhuaçu, MG (Google Earth Engine)
├── notebooks                   # Código fonte dos modelos e análise exploratória
│   └── eda.ipynb       # Análise Exploratória (EDA) e estatísticas
│   └── ml.ipynb        # Implementação e treinamento dos modelos de IA
└── TODO.md               # Gestão do projeto e tarefas pendentes
```

## Documentação

### Preparação do Ambiente

#### Instalar o Python  

1. Baixe e instale a versão mais recente do Python (recomendado 3.7 ou superior) no [site oficial do Python](https://www.python.org/).  
2. Durante a instalação, certifique-se de marcar a opção **"Add Python to PATH"**.  
3. Verifique as versões do Python e do `pip`  
   Certifique-se de que o Python e o `pip` foram instalados corretamente executando:  
    ```bash
    python --version
    pip --version
    ```

#### Criar um Ambiente Virtual (Opcional, mas Recomendado)  

Usar um ambiente virtual isola as dependências do projeto.

1. Abra um terminal ou prompt de comando.  
2. Navegue até a pasta do projeto.  
3. Execute o seguinte comando para criar um ambiente virtual:  
   ```bash
   python -m venv venv
   ```  
4. Ative o ambiente virtual:  
   - No Windows:  
     ```bash
     venv\Scripts\activate
     ```  
   - No macOS/Linux:  
     ```bash
     source venv/bin/activate
     ```

#### Instalar as Bibliotecas Necessárias

1. Instale o `pip` e o `setuptools` (se ainda não estiverem instalados)  
   Atualize o `pip` e o `setuptools` para a versão mais recente:  
    ```bash
    pip install --upgrade pip setuptools
    ```  
2. Instale as Bibliotecas  
   Use o `pip` para instalar as bibliotecas necessárias. Execute o comando:  
    ```bash
    pip install -r requirements.txt
    ```
   Mais detalhes sobre instalação do PyTorch: https://pytorch.org/get-started/locally/

   ## 📊 Análise Exploratória e Validação

Nesta sprint, o foco foi validar os modelos com dados reais históricos, avaliando:  
- **Desempenho preditivo com métricas (R², MAE, RMSE)**: MLP apresentou R²=0.825 e LSTM R²=0.702.  
- **Análise visual dos gráficos**: Foram gerados gráficos comparando as previsões dos modelos com os dados reais, destacando padrões sazonais, variações abruptas (ex: 2008) e comportamento específico de cada modelo.

🔍 **Estrutura do diretório validada:**

| Pasta                  | Última data de commit |
|------------------------|-----------------------|
| **GOOGLE_EARTH_ENGINE**| last week            |
| **PROCESSED**          | last week            |
| **SATVEG**             | last week            |
| **SIDRA**              | last week            |

---

## 🧠 Modelos Implementados e Avaliação

| Modelo | Descrição | Arquitetura | Métricas |
|--------|-----------|-------------|----------|
| MLP | Rede feed-forward para padrões não-lineares diretos | 32 → 16 neurônios, ReLU+Tanh, janela 5 obs. | R²=0.825, MAE=0.150, RMSE=0.210 |
| LSTM | Rede recorrente para dependências temporais longas | 2 camadas LSTM (32), janela 20 obs., dropout 20% | R²=0.702, MAE=0.195, RMSE=0.275 |

🔸 **Conclusão:** O MLP demonstrou melhor ajuste geral, enquanto o LSTM capturou melhor variações temporais. Ambos apresentam oportunidades de melhoria, incluindo variáveis climáticas e novos modelos.

## Equipe

### Membros (Grupo 25)

- Amandha Nery (RM560030) 
- Bruno Conterato (RM561048)
- Gustavo Castro (RM560831)
- Kild Fernandes (RM560615)
- Luis Emidio (RM559976)

### Professores

- Tutor: Leonardo Ruiz Orabona
- Coordenador: André Godoi

## Tecnologias Utilizadas

| Categoria              | Ferramentas                   |
|------------------------|-------------------------------|
| Linguagem              | Python 3.9+                   |
| Manipulação de Dados   | Pandas, NumPy                 |
| Visualização           | Matplotlib                    |
| Aprendizado Profundo   | PyTorch                       |
| Pré-processamento      | Scikit-learn (StandardScaler) |
| Ambiente               | Jupyter Notebook, CUDA (GPU)  |

## Contato

Se tiver alguma dúvida, sinta-se à vontade para entrar em contato. 🚀
