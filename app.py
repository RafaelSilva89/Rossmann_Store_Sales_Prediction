"""App Streamlit: previsão de vendas das próximas 6 semanas por loja e orçamento de reforma da rede Rossmann."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.append(str(RAIZ))

from src.config import ARQUIVO_LOJAS  # noqa: E402
from src.dados import carregar_lojas  # noqa: E402

st.set_page_config(page_title='Previsão de vendas Rossmann', page_icon='🏪', layout='wide')


@st.cache_data
def carregar_dados():
    previsoes = pd.read_csv(RAIZ / 'reports' / 'previsoes_6_semanas.csv', parse_dates=['data'])
    ficha = json.loads((RAIZ / 'Model' / 'modelo_rossmann_v2.json').read_text(encoding='utf-8'))
    return previsoes, carregar_lojas(ARQUIVO_LOJAS).set_index('loja'), ficha


def euros(valor):
    return '€ ' + f'{valor:,.0f}'.replace(',', '.')


previsoes, lojas, ficha = carregar_dados()
inferior, superior = ficha['faixa_total_6_semanas']
totais = previsoes.groupby('loja')['previsao'].sum().rename('previsto').to_frame()
totais['mínimo provável'] = totais['previsto'] * inferior
totais['máximo provável'] = totais['previsto'] * superior
periodo = f"{previsoes['data'].min():%d/%m/%Y} a {previsoes['data'].max():%d/%m/%Y}"

st.title('🏪 Quanto cada loja vai vender nas próximas 6 semanas?')
st.caption(f'Previsão de {periodo} para as {len(totais)} lojas da base pública Rossmann (Kaggle), em euros. '
           f'A faixa cobre o total real de {ficha["cobertura_faixa_teste"]:.0%} das lojas no teste.')

aba_loja, aba_reforma, aba_modelo = st.tabs(['🏪 Loja', '🧾 Orçamento de reforma', 'ℹ️ Sobre o modelo'])

with aba_loja:
    loja = st.selectbox('Loja', totais.index.tolist(), index=0)
    total = totais.loc[loja]
    col1, col2, col3 = st.columns(3)
    col1.metric('Vendas previstas (6 semanas)', euros(total['previsto']))
    col2.metric('Mínimo provável', euros(total['mínimo provável']))
    col3.metric('Máximo provável', euros(total['máximo provável']))
    cadastro = lojas.loc[loja]
    distancia = ('sem concorrente informado' if pd.isna(cadastro['distancia_concorrente'])
                 else f"concorrente a {cadastro['distancia_concorrente']:,.0f} m".replace(',', '.'))
    st.caption(f"Tipo de loja {cadastro['tipo_loja']} · sortimento {cadastro['sortimento']} · {distancia} · "
               f"{'participa' if cadastro['promo2'] == 1 else 'não participa'} da Promo2")

    diaria = previsoes[previsoes['loja'] == loja].sort_values('data')
    figura = go.Figure(go.Bar(x=diaria['data'], y=diaria['previsao'], marker_color='#1F4E79',
                              hovertemplate='%{x|%d/%m/%Y}: € %{y:,.0f}<extra></extra>'))
    figura.update_layout(height=320, margin=dict(t=10, b=10), yaxis_title='Venda prevista no dia (€)')
    st.plotly_chart(figura, use_container_width=True)
    st.caption(f"Dias fechados (domingos e feriados) aparecem com venda zero. "
               f"Dias com promoção: {int(diaria['promo'].sum())} de {len(diaria)}.")

with aba_reforma:
    st.subheader('Quanto reservar para a reforma de cada loja?')
    percentual = st.slider('Parcela do faturamento das 6 semanas destinada à reforma (%)', 1, 20, 5)
    conservador = st.toggle('Usar o mínimo provável (orçamento conservador)', value=False)
    base_calculo = totais['mínimo provável'] if conservador else totais['previsto']
    orcamento = (base_calculo * percentual / 100).rename('verba de reforma').to_frame()
    orcamento['vendas usadas no cálculo'] = base_calculo
    col1, col2 = st.columns(2)
    col1.metric('Verba total da rede', euros(orcamento['verba de reforma'].sum()))
    col2.metric('Verba média por loja', euros(orcamento['verba de reforma'].mean()))
    ranking = orcamento.sort_values('verba de reforma', ascending=False)
    st.dataframe(ranking.style.format(euros), use_container_width=True, height=420)
    st.download_button('Baixar a tabela (CSV)', ranking.round(2).to_csv().encode('utf-8'), 'orcamento_reforma.csv')

with aba_modelo:
    metricas = ficha['metricas_teste']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Erro do total por loja', f"{metricas['Erro do total por loja']:.1%}",
                help='Em média, quanto o total de 6 semanas previsto para uma loja erra o total real (teste)')
    col2.metric('Erro diário (MAPE)', f"{metricas['MAPE']:.1%}", help='Erro médio da venda de um dia, em %')
    col3.metric('Erro diário (MAE)', euros(metricas['MAE (€)']), help='Erro médio da venda de um dia, em euros')
    col4.metric('Referência: média da loja', f"{ficha['erro_total_referencia_teste']:.1%}",
                help='Erro do total por loja se a previsão fosse só a venda média de cada loja')
    st.markdown(f"**Modelo:** {ficha['modelo']}  \n**Treinado com:** {ficha['treinado_com']}")
    st.markdown('**Avisos:**\n' + '\n'.join(f'- {aviso}' for aviso in ficha['avisos']))
