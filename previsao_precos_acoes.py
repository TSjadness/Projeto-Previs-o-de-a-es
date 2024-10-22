import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os
from models.models import models_mlp

# Função para coleta dos dados de ações
def coletar_dados_acao(ticker, start_date, end_date):
    dados_acao = yf.download(ticker, start=start_date, end=end_date)
    return dados_acao['Close']

# Função de preprocessamento dos dados
def preprocessar_dados(dados, janela):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler.fit_transform(np.array(dados).reshape(-1, 1))

    X, y = [], []
    for i in range(janela, len(dados_normalizados)):
        X.append(dados_normalizados[i - janela:i, 0])
        y.append(dados_normalizados[i, 0])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler

# Função para salvar dados históricos
def salvar_historico_acoes(ticker, start_date, end_date):
    dados_historicos = yf.download(ticker, start=start_date, end=end_date)
    if not os.path.exists('data'):
        os.makedirs('data')  # Cria a pasta 'data' se não existir
    dados_historicos.to_csv('./data/historico_acoes.csv')

# Função para plotar gráficos com formatação de moeda
def plotar_graficos(y_teste_desnormalizado, y_pred_desnormalizado, moeda='USD'):
    # 1. Gráfico de Preço Real vs Previsto
    plt.figure(figsize=(14, 6))
    plt.plot(y_teste_desnormalizado, label='Preço Real - AAPL', color='blue')
    plt.plot(y_pred_desnormalizado, label='Preço Previsto - AAPL', linestyle='--', color='red')
    plt.title('Previsão de Preços da Ação AAPL com MLP')

    plt.xlabel('Dias')
    plt.ylabel(f'Preço de Fechamento ({moeda})')

    # Definir o formato da moeda ($ ou R$)
    fmt = '${x:,.2f}' if moeda != 'BRL' else 'R${x:,.2f}'  # Formato para Dólares ou Reais
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt.format(x=x)))

    plt.legend(loc='upper left')
    plt.grid(True)

    # Salvar gráfico de Preço Real vs Previsto
    plt.savefig('./images/previsao_precos.png')
    plt.show()  # Mostra o primeiro gráfico

    # 2. Erro Absoluto (diferença entre o real e o previsto)
    erro_absoluto = abs(y_teste_desnormalizado - y_pred_desnormalizado)
    plt.figure(figsize=(14, 4))
    plt.plot(erro_absoluto, label='Erro Absoluto', color='orange')
    plt.title('Possível Erro da Previsão de Preços da Ação AAPL')
    plt.xlabel('Dias')
    plt.ylabel(f'Erro Absoluto ({moeda})')

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt.format(x=x)))

    plt.legend(loc='upper left')
    plt.grid(True)

    # Salvar gráfico de Erro Absoluto
    plt.savefig('./images/erro_absoluto.png')
    plt.show()  # Mostra o segundo gráfico

    # 3. Exibir as métricas de desempenho
    mse = mean_squared_error(y_teste_desnormalizado, y_pred_desnormalizado)
    mae = mean_absolute_error(y_teste_desnormalizado, y_pred_desnormalizado)
    rmse = math.sqrt(mse)

    print(f'MSE: {mse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')

# Função principal que coordena o processo de previsão de preços
def previsao_precos_acoes(moeda='USD'):
    # 1. Coleta dos dados da ação
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    # Salvar histórico de ações
    salvar_historico_acoes(ticker, start_date, end_date)

    dados_acao = coletar_dados_acao(ticker, start_date, end_date)

    # 2. Preprocessamento dos dados
    janela_temporal = 5
    X, y, scaler = preprocessar_dados(dados_acao, janela_temporal)

    # 3. Separação dos dados em treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 4. Construção e treinamento do modelo MLP
    modelo = models_mlp(X_treino.shape[1])
    modelo.fit(X_treino, y_treino, epochs=50, batch_size=32, verbose=1)

    # 5. Predição e avaliação do modelo
    y_pred = modelo.predict(X_teste)
    y_teste_desnormalizado = scaler.inverse_transform(y_teste.reshape(-1, 1))
    y_pred_desnormalizado = scaler.inverse_transform(y_pred)

    # 6. Plotar os gráficos
    plotar_graficos(y_teste_desnormalizado, y_pred_desnormalizado, moeda=moeda)

# Chamando a função principal
if __name__ == '__main__':
    previsao_precos_acoes(moeda='BRL')  # Pode mudar para 'USD' ou 'BRL'
