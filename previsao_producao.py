import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="Sistema de Análise e Previsão de Séries Temporais", layout="wide")

st.title("Sistema Previsão de Produção de Leite.")

# Instruções para o usuário
st.markdown("""
### Instruções de Uso:
1. Escolha o período inicial da série temporal.
2. Informe quantos meses de previsão você deseja (máximo de 48 meses).
3. Clique no botão "Processar" para gerar os gráficos.
   
**Obs:** É possível expandir os gráficos para melhor visualização.
""")

# Carregar o CSV automaticamente
csv_file = 'monthly-milk-production-pounds-p.csv'
data = pd.read_csv(csv_file, header=None)

data_inicio = date(2024, 5, 1)
periodo = st.sidebar.date_input("Período Inicial da Série", data_inicio)
periodo_previsao = st.sidebar.number_input("Informe quantos meses quer prever", min_value=1, max_value=48, value=12)
processar = st.sidebar.button("Processar")

# Decomposição e gráficos
if processar:
    try:
        ts_data = pd.Series(data.iloc[:, 0].values, index=pd.date_range(
            start=periodo, periods=len(data), freq='ME'
        ))  # o :,0 vai pegar os dados da primeira coluna 

        decomposicao = seasonal_decompose(ts_data, model='additive')
        fig_decomposicao = decomposicao.plot()
        fig_decomposicao.set_size_inches(10, 8)

        modelo = SARIMAX(ts_data, order=(2, 0, 0), seasonal_order=(0, 1, 1, 12))
        modelo_fit = modelo.fit()

        previsao = modelo_fit.get_forecast(steps=periodo_previsao)
        previsao_index = pd.date_range(start=ts_data.index[-1], periods=periodo_previsao + 1, freq='M')[1:]
        previsao_series = pd.Series(previsao.predicted_mean, index=previsao_index)

        fig_previsao, ax = plt.subplots(figsize=(10, 5))
        ts_data.plot(ax=ax)  # dados originais
        previsao_series.plot(ax=ax, style='r--')  # previsão, num mesmo gráfico vermelho tracejado

        col1, col2 = st.columns([3, 3])
        with col1:
            st.write("Decomposição, Tendência, Sazonalidade e Resíduo")
            st.pyplot(fig_decomposicao)
        with col2:
            st.write("Gráfico Previsão de Produção")
            st.pyplot(fig_previsao)
        
        st.write("Dados da Previsão")
        st.dataframe(previsao_series)

        # Mostrar informações de otimização
        output = modelo_fit.summary().as_text()
        st.markdown("### Informações de Otimização do Modelo SARIMAX")
        st.text(output)

        st.markdown("""
        ### Explicação dos Resultados:
        - **Coeficiente AR(L1) e AR(L2)**: Representam a autocorrelação da série temporal nos lags 1 e 2. Valores próximos de 1 ou -1 indicam forte influência dos valores passados nos valores futuros. Valores próximos de 0 indicam pouca influência.
        - **Coeficiente MA.S(L12)**: Representa a média móvel sazonal no lag 12. Valores próximos de -1 indicam que um valor alto em um mês é seguido por um valor baixo após 12 meses, enquanto valores próximos de 0 indicam pouca influência sazonal.
        - **Sigma²**: Variância dos resíduos do modelo. Valores menores indicam um ajuste mais preciso do modelo, geralmente quanto menor, melhor.
        - **Ljung-Box (L1) (Q)**: Teste para a autocorrelação dos resíduos. Valores de p acima de 0.05 indicam que não há autocorrelação significativa nos resíduos.
        - **Jarque-Bera (JB)**: Teste de normalidade dos resíduos. Valores de p abaixo de 0.05 indicam que os resíduos não seguem uma distribuição normal.
        - **Heteroskedasticity (H)**: Teste para heterocedasticidade dos resíduos. Valores de p acima de 0.05 indicam que os resíduos têm variância constante ao longo do tempo.
        - **AIC (Akaike Information Criterion)**: Critério de informação de Akaike. Valores mais baixos indicam um modelo melhor ajustado.
        - **BIC (Bayesian Information Criterion)**: Critério de informação Bayesiano. Assim como o AIC, valores mais baixos indicam um modelo melhor ajustado.
        """)
    

        # Mostrar informações adicionais após processamento
        st.markdown("""
        ### Descrição do Dataset:
        Este dataset contém a produção mensal de leite em libras por vaca leitera.
        """)

        st.markdown("""
        ### Interpretação dos Resultados:
        - **Decomposição**: O gráfico de decomposição mostra as partes observadas, tendência, sazonalidade e resíduo da série temporal.
        - **Previsão**: O gráfico de previsão mostra os dados históricos em azul e as previsões futuras em vermelho tracejado.
        """)

        st.markdown("""
        ### Limitações do Modelo:
        Este modelo assume que os padrões históricos continuarão no futuro. Mudanças abruptas ou eventos inesperados podem não ser previstos com precisão.
        """)

    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")
