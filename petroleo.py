import util
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from datetime import timedelta

#--------------------------------------------------------------------------------------------
# TRATANDO OS DADOS
#--------------------------------------------------------------------------------------------

lst_preco = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view', attrs={'id': 'grd_DXMainTable'}, decimal=',', thousands='.')

df_preco = pd.DataFrame(lst_preco[0])
df_preco.rename(columns={df_preco.columns[0]: 'data', df_preco.columns[1]: 'preco'}, inplace=True)
df_preco.drop(index=0, inplace=True)
df_preco.reset_index(drop=True, inplace=True)

# Converter a coluna 'Data' para o formato datetime e a coluna 'Preço' para numeric
df_preco['data'] = pd.to_datetime(df_preco['data'], format='%d/%m/%Y')
df_preco['preco'] = pd.to_numeric(df_preco['preco'])

# Ordenar o DataFrame pela data, se necessário
df_preco = df_preco.sort_values(by='data', ascending=True)

#--------------------------------------------------------------------------------------------
# > EXTRAINDO NOVAS INFORMACOES
#--------------------------------------------------------------------------------------------

df_preco['variacao'] = df_preco['preco'].diff()
df_preco['variacao'] = df_preco['variacao'].fillna(0)

df_preco['variacao_percentual'] = (df_preco['preco'].pct_change() * 100).round(2)
df_preco['variacao_percentual'] = df_preco['variacao_percentual'].fillna(0)

#--------------------------------------------------------------------------------------------
# EXIBINDO DADOS INICIAIS
#--------------------------------------------------------------------------------------------

st.set_page_config(
  page_title='FIAP - Petróleo com ML',
  layout='wide'
)
st.header('Previsão do preço do petróleo com ML')
st.write('## Dados iniciais')
st.write(f'No momento de escrita deste trabalho os dados mais recentes são de {df_preco["data"].max().strftime("%d-%m-%Y")}')
df_preco_exibicao = df_preco.copy()
#df_preco_exibicao['data'] = df_preco_exibicao['data'].dt.strftime('%d-%m-%Y')
df_preco_exibicao = df_preco_exibicao.rename(columns={'data':'Data', 
                                                      'preco':'Preço (US$)', 
                                                      'variacao':'Variação', 
                                                      'variacao_percentual': 'Variação %'})

data_inicio_tabela, data_fim_tabela = st.date_input(
  ":blue[Selecione o período:]",
  [df_preco_exibicao['Data'].min(),
  df_preco_exibicao['Data'].max()]
)
df_filtrado = df_preco_exibicao[(df_preco_exibicao["Data"] >= pd.to_datetime(data_inicio_tabela)) & (df_preco_exibicao["Data"] <= pd.to_datetime(data_fim_tabela))]

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para o período selecionado.")
else:
    st.success("Dados filtrados com sucesso!")
    st.dataframe(df_filtrado)

col1_variacao, col2_variacao = st.columns(2)
with col1_variacao:
  st.markdown(f':orange[Variação média:] {df_filtrado["Variação"].mean():.2f}')
with col2_variacao:
  st.markdown(f':orange[Variação % média:] {df_filtrado["Variação %"].mean():.2f}')  

#--------------------------------------------------------------------------------------------
# SEPARANDO DADOS PARA ML E TREINANDO MODELOS
#--------------------------------------------------------------------------------------------

st.write('## Análise dos dados')
# Criando um DataFrame para armazenar a performance de treinamento de cada modelo.
# A ideia também seria armazenar e comparar no final a performance para vários períodos de previsão mas precisaria de mais tempo de codificação
if "performance_modelos" not in st.session_state:
    st.session_state.performance_modelos = pd.DataFrame(columns=['Modelo', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'Acertividade'])

# Definindo uma janela para ser utilizada na previsão
janela = 5

# removendo as colunas que foram utilizadas para o dashboard no powerBi
df_preco.drop(columns=['variacao', 'variacao_percentual'], axis=1, inplace=True)

# Realizando a cópia dos dados para um novo DataFrame para termos um backup dos originais antes de começar a transfeormar mais coisas

data_inicial = '2024-03-24'
preco_2024_2025 = df_preco.loc[df_preco['data'] >= data_inicial].copy()
preco_2024_2025.rename(columns={'data': 'ds', 'preco': 'y'}, inplace=True)
preco_2024_2025.set_index('ds', inplace=True)

util.calcular_ma_std(preco_2024_2025, janela)
st.write('Iremos utilizar os dados de aproximdamente um ano pois visualmente não encontramos uma sazonalidade na oscilação dos \
  preços (como proposta de melhoria, estes dados poderiam ser analisados de forma dinâmica, trazendo outras perspectivas para a análise).\
  \nComo primeiro passo, vamos olhar para a média móvel e desvio padrão neste período.')
st.pyplot(util.plot_ma_std(preco_2024_2025, janela))

#------------------------------------------------------------------------------------
st.title('1. Testando a estacionariedade')
#------------------------------------------------------------------------------------
st.write(util.testar_estacionariedade(preco_2024_2025["y"].values))

st.write('## 1.1 Tentando estacionar a série')
st.write('Para tentar estacionar a série, vamos aplicar uma função logarítmica no dataframe, em seguida vamos calcular uma nova média móvel. \
  Este valor será utilizado para subitrair do resultado da operação logarítimica. Por fim, uma nova média móvel será  analisada')

# Aplicando log na função
preco_2024_2025_log = pd.DataFrame()
preco_2024_2025_log['y'] = np.log(preco_2024_2025['y'])

# Claculando uma nova média móvel
util.calcular_ma_std(preco_2024_2025_log, janela)

# Subtraindo a média móvel
preco_2024_2025_s = pd.DataFrame()
preco_2024_2025_s['y'] = (preco_2024_2025_log['y'] - preco_2024_2025_log['ma']).dropna()
preco_2024_2025_s.dropna(inplace=True)

# Claculando uma nova média móvel
util.calcular_ma_std(preco_2024_2025_s, janela)

# Analisando visualmente as mudanças
st.pyplot(util.plot_ma_std(preco_2024_2025_s, janela))

st.write('Com este novo resultado, vamos diferenciar a série e calcular sua estacionariedade. Levando ao gráfico:')
# Diferenciando a série
preco_2024_2025_diff = util.aplicar_diferenciacao(preco_2024_2025_s)
# Claculando uma nova média móvel
util.calcular_ma_std(preco_2024_2025_diff, janela)
# Analisando visualmente as mudanças
st.pyplot(util.plot_ma_std(preco_2024_2025_diff, janela))

st.write(util.testar_estacionariedade(preco_2024_2025_diff['y'].values))
st.write('Com a diferenciação o p-valor ficou muito próximo a zero e isso sugere que a série temporal foi excessivamente transformada')

#------------------------------------------------------------------------------------
st.title('2. Treinando modelos')
#------------------------------------------------------------------------------------

st.write('# 2.1 ARIMA')
st.write('## 2.1.1 Teste ACF/PACF')
st.write('Não conseguimos estacionar a série, vamos então trabalhar com os dados iniciais')
preco_2024_2025.drop(columns=['ma', 'std'], axis=1, inplace=True)

st.write('Para o primeiro modelo escolhemos o ARIMA.\
  \nComo hiper-parâmetros precisamos definir p e q, os quais podemos obter, ou inferir, a partir dos testes ACF e PACF')

fig, axes = plt.subplots(2, 1, figsize=(20, 8))

# Após realizar alguns testes com lags diferentes, chegamos nos valores abaixo
plot_acf(preco_2024_2025['y'], ax=axes[0], lags=120, title="Função de Autocorrelação (ACF)")
plot_pacf(preco_2024_2025['y'], ax=axes[1], lags=10, title="Função de Autocorrelação Parcial (PACF)", method="ywm")
st.pyplot(fig)

st.write('## 2.1.2 Análise do modelo')

teste_2024_20245 = df_preco.iloc[-10:].copy()
teste_2024_20245.rename(columns={'data': 'ds', 'preco': 'y'}, inplace=True)
teste_2024_20245.set_index('ds', inplace=True)

treino_2024_2025 = df_preco.query("data >= @data_inicial").iloc[:-10].copy()
treino_2024_2025.rename(columns={'data': 'ds', 'preco': 'y'}, inplace=True)
treino_2024_2025.set_index('ds', inplace=True)

st.write(f'Treino: de {treino_2024_2025.index.min().strftime("%d-%m-%Y")} a {treino_2024_2025.index.max().strftime("%d-%m-%Y")} - {treino_2024_2025.shape[0]} dias \
  \nTeste: de {teste_2024_20245.index.min().strftime("%d-%m-%Y")} a {teste_2024_20245.index.max().strftime("%d-%m-%Y")} - {teste_2024_20245.shape[0]} dias')

compilado_2024_2025 = pd.DataFrame({'ds': teste_2024_20245.index, 'y': teste_2024_20245.values.ravel()})

col1_arima, col2_arima, col3_arima = st.columns(3)
with col1_arima:
  input_p = st.number_input(':blue[Valor de "p"]', 0, 10, 2)
with col2_arima:
  input_d = st.number_input(':blue[Valor de "d"]', 0, 5, 1)
with col3_arima:
  input_q = st.number_input(':blue[Valor de "q"]', 0, 120, 104)

if st.button('Processar ARIMA'):
    with st.status("Isso pode demorar um pouco. Processando... ⏳", expanded=True) as status:
        # passando os dados do PACF/ACF
        modelo = ARIMA(treino_2024_2025.values, order=(input_p, input_d, input_q)) # Ajustar os parâmetros (p, d, q)
        arima_2024_2025 = modelo.fit()
        # Previsões no conjunto de teste
        arima_y_pred = arima_2024_2025.forecast(steps=len(teste_2024_20245))
        
        
        # Criando novo DF para facilitar a visualização e comparação dos dados dos outros modelos
        #compilado_2024_2025 = pd.DataFrame({'ds': teste_2024_20245.index, 'y': teste_2024_20245.values.ravel(), 'y_pred_arima': arima_y_pred})
        compilado_2024_2025['y_pred_arima'] = arima_y_pred
        
        
        # Criando novo DF para facilitar a visualização e comparação dos dados dos outros modelos
        compilado_2024_2025['y_pred_arima'] = arima_y_pred
        resultado, st.session_state.performance_modelos = util.calcular_métricas(compilado_2024_2025['y'], compilado_2024_2025['y_pred_arima'], st.session_state.performance_modelos, 'ARIMA_2024_2025')
        st.write(resultado)
        st.pyplot(util.plot_previsao(treino_2024_2025.index,
                    treino_2024_2025.values,
                    compilado_2024_2025['ds'],
                    compilado_2024_2025['y'],
                    compilado_2024_2025['y_pred_arima'],
                    'ARIMA')
                )
        status.update(label="Processamento concluído ✅", state="complete")

st.write('# 2.2 Holt Winters')
st.write('## 2.2.1 Escolha de parâmetros')
# Vamos aplicar um novo modelo de treinamento
# Dados de treino e teste
treino = treino_2024_2025.values
teste = teste_2024_20245.values

# Inicializando variáveis para armazenar os resultados
melhor_periodo = None
melhor_rmse = float('inf')
resultados = []

st.write('Para tentar melhorar sua performance, vamos rodar aproximadamente 60 testes usando "ExponentialSmoothing" e escolher o melhor (menor RMSE)')


if st.button('Processar Holt-Winters'):
  with st.status("Isso pode demorar um pouco. Processando... ⏳", expanded=True) as status:
    
    for periodo in range(2, 60):
      # Ajustando o modelo
      modelo = ExponentialSmoothing(treino, trend='add', seasonal='add', seasonal_periods=periodo)
      modelo_ajustado = modelo.fit()

      # Fazendo previsões
      previsoes = modelo_ajustado.forecast(len(teste))

      # Calculando RMSE
      rmse = mean_squared_error(teste, previsoes)
      resultados.append((periodo, rmse))

      # Atualizando o melhor período
      if rmse < melhor_rmse:
          melhor_rmse = rmse
          melhor_periodo = periodo

    # Criando um DataFrame com os resultados
    df_resultados = pd.DataFrame(resultados, columns=['Seasonal_Period', 'RMSE'])

    # Exibindo o melhor resultado
    st.markdown(f":orange[Melhor Seasonal_Period:] {melhor_periodo}, com :orange[RMSE:] {melhor_rmse}")
    st.write(df_resultados)
    
    
    # Com base nos testes anteriores, vamos utilizar o parâmetro de melhor
    # desempenho, vamos treinar, prever e armazenar o resultado deste modelo
    compilado_2024_2025['y_pred_hw'] = ExponentialSmoothing(treino_2024_2025.values, 
                                                            trend='add', 
                                                            seasonal='add', 
                                                            seasonal_periods=24).fit().forecast(len(teste_2024_20245.values))

    st.write('Treinando o modelo com os dados selecionados temos:')
    resultado, st.session_state.performance_modelos = util.calcular_métricas(compilado_2024_2025['y'], compilado_2024_2025['y_pred_hw'], st.session_state.performance_modelos, 'Holt-Winters_2024_2025')
    st.write(resultado)
    st.pyplot(util.plot_previsao(treino_2024_2025.index,
                treino_2024_2025.values,
                compilado_2024_2025['ds'],
                compilado_2024_2025['y'],
                compilado_2024_2025['y_pred_hw'],
                'Holt-Winters')
            )
    status.update(label="Processamento concluído ✅", state="complete")

st.write('# 2.3 Prophet')
st.write('Para este modelo, não temos tantos hiperparâmetros para serem passados')
# Preparando os dados de treino e teste para o modelo, sempre considerando o base dos anteriores
treino_2024_2025_ppt = pd.DataFrame()
treino_2024_2025_ppt['ds'] = treino_2024_2025.index
treino_2024_2025_ppt['y'] = treino_2024_2025.values

teste_2024_2025_ppt = pd.DataFrame()
teste_2024_2025_ppt['ds'] = teste_2024_20245.index
teste_2024_2025_ppt['y'] = teste_2024_20245.values

if st.button('Processar Prophet'):
  with st.status("Isso pode demorar um pouco. Processando... ⏳", expanded=True) as status:
    # Treinar o modelo
    modelo_ppt_2024_2025 = Prophet(interval_width=0.90, daily_seasonality=True)
    modelo_ppt_2024_2025.fit(treino_2024_2025_ppt)

    # Gerar datas futuras
    previsao_ppt_2019_2023 = modelo_ppt_2024_2025.make_future_dataframe(periods=len(teste_2024_2025_ppt))

    # Fazer a previsão
    previsoes_ppt_2019_2023 = modelo_ppt_2024_2025.predict(previsao_ppt_2019_2023)
    previsoes_ppt_2019_2023['ds'] = previsoes_ppt_2019_2023['ds'].dt.date

    # Armazenar as previsões do modelo
    compilado_2024_2025['y_pred_ppt'] = previsoes_ppt_2019_2023['yhat']

    resultado, st.session_state.performance_modelos = util.calcular_métricas(compilado_2024_2025['y'], compilado_2024_2025['y_pred_ppt'], st.session_state.performance_modelos, 'Prophet_2024_2025')
    st.write(resultado)
    st.pyplot(util.plot_previsao(treino_2024_2025_ppt['ds'],
                treino_2024_2025_ppt['y'],
                compilado_2024_2025['ds'],
                compilado_2024_2025['y'],
                compilado_2024_2025['y_pred_ppt'],
                'Prophet_2024_2025'))
    
    status.update(label="Processamento concluído ✅", state="complete")

#------------------------------------------------------------------------------------
st.title('3. Prevendo dados futuros')
#------------------------------------------------------------------------------------
st.write('# 3.1 Escolhendo modelo')
st.write('Compilado dos processamentos:')

if st.button('Exibir dados'):
    # Vamos copiar os dados de desempenho para um novo DataFrame
    # Isso para podermos realizar algumas manipulações sem quebrar o original caso desejemos rodar novamente algum teste passado
    performance_final = st.session_state.performance_modelos.copy()
    performance_final.set_index('Modelo',  inplace=True)
    st.write(performance_final.sort_values(by='Acertividade', ascending=False))

st.write('# 3.2 Aplicando o modelo')
st.write('Neste trabalho utilizamos como parâmetro de escolha a acertividade. Consultando a tabela anterior, o \
  modelo Holt-Winters será o escolhido para realisar a previsão de dados que não temos. Para isto iremos juntar \
  os dados de treino e teste para serem o treino e o "teste" será a previsão')

if st.button('Realizar previsão'):
  with st.status("Isso pode demorar um pouco. Processando... ⏳", expanded=True) as status:
    # utilizando os dados do último ano
    df_treino_final = pd.DataFrame()
    df_treino_final['ds'] = df_preco.loc[(df_preco['data'] >= data_inicial)]['data']
    df_treino_final['y'] = df_preco.loc[(df_preco['data'] >= data_inicial)]['preco']

    modelo = ExponentialSmoothing(df_treino_final['y'], trend='add', seasonal='add', seasonal_periods=24)
    modelo_ajustado = modelo.fit()

    # Prevendo os proximos 5 valores
    previsao = modelo_ajustado.forecast(5)
    st.write('valores previstos:')
    st.write(previsao)

    # Criando possíveis datas para os valores previstos
    data_inicial = df_treino_final['ds'].max() + timedelta(days=1)
    datas_previsao = pd.date_range(start=data_inicial, periods=len(previsao), freq="D")

    st.write('As datas exibidas não está considerando se dia útil assim como vieram os dados utilizados até o momento')
    st.pyplot(util.plot_previsao(
        df_treino_final['ds'],
        df_treino_final['y'],
        datas_previsao,
        previsao,
        previsao,
        'Holt-Winters'))
    status.update(label="Processamento concluído ✅", state="complete")