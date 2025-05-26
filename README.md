# ec-ingredion-3

> Este projeto faz parte do curso de **Inteligência Artificial** oferecido pela [FIAP](https://github.com/fiap) - Online 2024. Este repositório reúne os materiais relacionados ao **Enterprise Challenge - Ingredion**, correspondendo à **Sprint 3** do desafio.

- Notebook para Machine Learning: [Jupyter Notebook](./notebooks/ml.ipynb)
- Notebook de Análise Exploratória de Dados: [Jupyter Notebook](./notebooks/eda.ipynb)
- Relatório Técnico: [Documento PDF](./report.pdf)

## Descrição

Este repositório contém o código-fonte e a documentação da SPRINT 3 do Challenge Ingredion, focada na validação de um modelo de Inteligência Artificial (IA) para previsão de produtividade agrícola. O objetivo principal é correlacionar as previsões de produtividade do modelo com dados reais históricos, avaliando sua confiabilidade e precisão.

## Estrutura de Arquivos

O repositório está organizado da seguinte forma:

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
├── report.pdf            # Relatório técnico
├── requirements.txt      # Lista de dependências do projeto
├── scripts               # Notebooks para extração e preparação dos dados
│   └── extract-analysis-data.ipynb           # Preparação e integração dos dados para análise
│   └── extract-ndvi-manhuacu.ipynb # Extração NDVI de Manhuaçu, MG (Google Earth Engine)
├── notebooks                   # Código fonte dos modelos e análise exploratória
│   └── eda.ipynb       # Análise Exploratória (EDA) e estatísticas
│   └── ml.ipynb        # Implementação e treinamento dos modelos de IA
└── TODO.md               # Gestão do projeto e tarefas pendentes
```

## Instruções para Execução

Para executar os notebooks, siga as instruções abaixo:

### Clone o Repositório

```bash
git clone https://github.com/luisfuturist/ec-ingredion-3.git
cd ec-ingredion-3
```

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

### Observações

**Observações:**

*   É necessário ter uma conta no Google Earth Engine para executar os scripts de extração dos dados NDVI.
*   Os dados brutos do IBGE (arquivos Excel) já estão incluídos no repositório, mas podem ser atualizados baixando os dados mais recentes diretamente do site do IBGE.

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

Se tiver alguma dúvida, sinta-se à vontade para entrar em contato.
