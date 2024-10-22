# Previsão de Preços de Ações com Redes Neurais Multilayer Perceptron (MLP)

Este projeto utiliza uma Rede Neural Artificial (Multilayer Perceptron - MLP) para prever os preços futuros de ações com base em dados históricos. Ele foi desenvolvido como parte da avaliação prática da disciplina de Inteligência Artificial no curso de Sistemas de Informação.

## 1. Objetivo

A aplicação prevê o preço de fechamento de uma ação (neste exemplo, da Apple - AAPL) com base em uma série temporal dos preços de fechamento anteriores. O projeto utiliza uma rede neural MLP para realizar previsões a partir de uma janela de dias anteriores, buscando prever o próximo valor.

## 2. Tecnologias Utilizadas

O projeto foi desenvolvido utilizando a linguagem **Python** e as seguintes bibliotecas:

- **yfinance**: Para coleta de dados históricos de ações.
- **pandas**: Para manipulação de dados.
- **numpy**: Para cálculos numéricos.
- **scikit-learn**: Para pré-processamento dos dados.
- **tensorflow/keras**: Para construção e treinamento do modelo de rede neural.
- **matplotlib**: Para visualização gráfica dos resultados.

## 3. Estrutura do Projeto

- **previsao_precos_acoes.py**: Arquivo principal contendo o código do projeto.
- **README.md**: Este arquivo, explicando o projeto e como executá-lo.
- **/images**: Pasta para armazenar gráficos gerados pelo código.
- **/data**: Pasta opcional para armazenar dados de entrada (preços de ações, caso você deseje baixar manualmente).

## 4. Como Executar o Projeto

### 4.1. Pré-requisitos

Antes de rodar o projeto, você precisará instalar as bibliotecas necessárias. Utilize o seguinte comando no terminal para instalar as dependências:

```bash
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib
