import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

#--------------------------------------------------------------------------------------------
# DEFININDO METODOS
#--------------------------------------------------------------------------------------------

#Função para calcular a média móvel e o desvio padrão
#:param df: DataFrame com a coluna 'y'
#:param window: Janela para calcular a média móvel
def calcular_ma_std(df, window):
    df['ma'] = df['y'].rolling(window=window).mean()
    df['std'] = df['y'].rolling(window=window).std()

#Função para plotar a média móvel e o desvio padrão, bem como nosso dado target
#:param df: DataFrame com a coluna 'y' e as colunas 'ma' e 'std'
#:param window: Janela para exibir no label a quantidade de dias
def plot_ma_std(df, window):
  fig, ax = plt.subplots(figsize=(16, 6))
  
  ax.plot(df.index, df['y'], color='blue', label='Original', alpha=0.5)
  ax.plot(df.index, df['ma'], color='red', label=f'Média Móvel ({window} dias)', alpha=0.6)
  ax.fill_between(
      df.index,
      df['ma'] - df['std'],
      df['ma'] + df['std'],
      color='green',
      alpha=0.2,
      label='Desvio Padrão'
  )

  # Configraundo detalhes do gráfico
  ax.legend(loc='best')
  ax.set_title(f'Média Móvel e Desvio Padrão ({df.index.min().strftime("%d-%m-%Y")} - {df.index.max().strftime("%d-%m-%Y")})')
  ax.set_xlabel('Índice')
  ax.set_ylabel('Valor')
  ax.grid(True)
  
  return fig

#Função para testar a estacionariedade da série
#:param serie_valores: Valores da série temporal
def testar_estacionariedade(serie_valores):
    result = adfuller(serie_valores)
    estatistico_teste, p_valor, _, _, valores_criticos, _ = result

    partes = [
        "Teste ADF - Sendo o teste estatístico e o p-valor maiores que os valores críticos, esta série NÃO é estacionária.",
        f"Para nosso teste de estacionariedade, vamos considerar um p-valor de 0.05, para rejeitar H0 com 95% de confiança.",
        f"\nTeste estatístico: {estatistico_teste:.7f}",
        f"\nP-Value: {p_valor:.7f}",
        "\nValores críticos:"
    ]

    partes += [f"\n\t{nivel}: {valor:.4f}" for nivel, valor in valores_criticos.items()]
    partes.append("Considerando a margem de 5%, temos que:")

    if p_valor > valores_criticos['5%'] and estatistico_teste > valores_criticos['5%']:
        partes.append("p-valor e teste estatístico são maiores que o limite definido. Logo, esta série NÃO é estacionária.")
    elif p_valor > valores_criticos['5%']:
        partes.append("p-valor continua maior que o limite definido.")
    elif estatistico_teste > valores_criticos['5%']:
        partes.append("teste estatístico continua maior que o limite definido.")
    else:
        partes.append("p-valor e teste estatístico são menores que o limite definido.")

    return "\n".join(partes)


#Função para aplicar a diferenciação
#:param df: DataFrame com a coluna 'y'
#:return: DataFrame após a diferenciação
def aplicar_diferenciacao(df):
  df_diff = pd.DataFrame()
  df_diff['y'] = df['y'].diff(1).dropna()
  df_diff.dropna(inplace=True)
  return df_diff
      
#Função para plotar dois gráficos. Um contendo os dados de treino e o outro com os dados de teste e previsão
#:param s_ds_treino: pd.Series contendo as datas de treino
#:param s_treino: pd.Series contendo os valores de treino
#:param s_ds: pd.Series contendo as datas de teste
#:param s_y: pd.Series contendo os valores de teste
#:param s_y_pred: pd.Series Valores de previsão
#:param metodologia: String com o nome do modelo utilizado
def plot_previsao(s_ds_treino, s_treino, s_ds, s_y, s_y_pred, metodologia):
  fig, axes = plt.subplots(2, 1, figsize=(15, 8))
  qt_dias = len(s_y_pred)

  axes[0].plot(s_ds_treino, s_treino, label='Valores Reais', color='blue', alpha=0.5)
  axes[0].plot(s_ds, s_y, label='Dados de teste', color='green', alpha=0.7)
  axes[0].plot(s_ds, s_y_pred, label=f'Previsão ({qt_dias} dias)', color='red', alpha=0.7)
  axes[0].set_title(f'Previsão usando {metodologia} - Dados completos')
  axes[0].set_ylabel('Valores')
  axes[0].legend()
  axes[0].grid()

  axes[1].plot(s_ds, s_y, label='Dados de teste', color='green', alpha=0.7)
  axes[1].plot(s_ds, s_y_pred, label=f'Previsão ({qt_dias} dias)', marker='^', linestyle='--', color='red', alpha=0.7)
  axes[1].set_title(f'Previsão usando {metodologia} - Apenas teste e previsão')
  axes[1].set_ylabel('Valores')
  axes[1].legend()
  axes[1].grid()  
  
  return fig      


#Função para calcular as métricas de avaliação
#:param y_true: pd.Series com valores reais
#:param y_pred: pd.Series com valores previstos
#:param nome_modelo: Nome do modelo
def calcular_métricas(y_true, y_pred, performance_modelos, nome_modelo=''):
  # Asumindo 'y_true' sendo os valores reais e 'y_pred' as previsões do modelo
  metrica = []

  # Erro Médio Absoluto (MAE)
  mae = mean_absolute_error(y_true, y_pred)
  metrica.append(('MAE:', mae))

  # Erro Quadrático Médio (MSE)
  mse = mean_squared_error(y_true, y_pred)
  metrica.append(('MSE:', mse))

  # Raiz do Erro Quadrático Médio (RMSE)
  rmse = np.sqrt(mse)
  metrica.append(('RMSE:', rmse))

  # Erro Percentual Absoluto Médio (MAPE)
  mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
  #mape = Decimal(f"{round(mape, 2):.2f}")
  mape = round(mape, 2)
  metrica.append(('MAPE', f"{mape}%"))

  # Coeficiente de Determinação (R²)
  r2 = r2_score(y_true, y_pred)
  metrica.append(('R²:', r2))

  # Atualiza DataFrame se fornecido
  if nome_modelo != '':
    # Verifica se o modelo já existe no DataFrame
    if nome_modelo in performance_modelos['Modelo'].values:
        # Atualiza a linha existente
        performance_modelos.loc[performance_modelos['Modelo'] == nome_modelo, ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'Acertividade']] = [mae, mse, rmse, mape, r2, 100 - mape]
    else:
        # Adiciona uma nova linha
        performance_modelos.loc[len(performance_modelos)] = [nome_modelo, mae, mse, rmse, mape, r2, 100 - mape]

  return metrica, performance_modelos