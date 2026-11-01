"""App Streamlit: previsão de vendas das próximas 6 semanas por loja, orçamento de reforma e monitoramento do modelo."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.append(str(RAIZ))

from src.avaliacao import cenarios_por_loja  # noqa: E402
from src.config import ARQUIVO_LOJAS  # noqa: E402
from src.dados import carregar_lojas  # noqa: E402
from src.monitoramento import painel_semanas, relatorio_drift  # noqa: E402

st.set_page_config(page_title='Previsão de vendas Rossmann', page_icon='🏪', layout='wide')


@st.cache_data
def carregar_dados():
    previsoes = pd.read_csv(RAIZ / 'reports' / 'previsoes_6_semanas.csv', parse_dates=['data'])
    ficha = json.loads((RAIZ / 'Model' / 'modelo_rossmann_v2.json').read_text(encoding='utf-8'))
    efeitos = pd.read_csv(RAIZ / 'reports' / 'shap_lojas.csv', index_col='loja')
    return previsoes, carregar_lojas(ARQUIVO_LOJAS).set_index('loja'), ficha, efeitos


@st.cache_data
def carregar_monitoramento():
    teste = pd.read_csv(RAIZ / 'reports' / 'monitoramento_teste.csv.gz', parse_dates=['data'])
    referencia = pd.read_csv(RAIZ / 'reports' / 'referencia_monitoramento.csv.gz')
    return teste, referencia


def euros(valor):
    return '€ ' + f'{valor:,.0f}'.replace(',', '.')


def pct(valor, casas=1):
    return f'{valor:.{casas}%}'.replace('.', ',')


previsoes, lojas, ficha, efeitos = carregar_dados()
inferior, superior = ficha['faixa_total_6_semanas']
totais = cenarios_por_loja(previsoes['loja'], previsoes['previsao'], (inferior, superior))
periodo = f"{previsoes['data'].min():%d/%m/%Y} a {previsoes['data'].max():%d/%m/%Y}"

st.title('🏪 Quanto cada loja vai vender nas próximas 6 semanas?')
st.caption(f'Previsão de {periodo} para as {len(totais)} lojas da base pública Rossmann (Kaggle), em euros. '
           f'No teste, o total real ficou entre o pior e o melhor cenário em {ficha["cobertura_faixa_teste"]:.0%} das lojas.')

aba_loja, aba_cenarios, aba_reforma, aba_monitoramento, aba_modelo = st.tabs(
    ['🏪 Loja', '📊 Cenários', '🧾 Orçamento de reforma', '📡 Monitoramento', 'ℹ️ Sobre o modelo'])

with aba_loja:
    loja = st.selectbox('Loja', totais.index.tolist(), index=0)
    total = totais.loc[loja]
    col1, col2, col3 = st.columns(3)
    col1.metric('Vendas previstas (6 semanas)', euros(total['cenário provável']), help='Cenário provável: o total previsto')
    col2.metric('Pior cenário', euros(total['pior cenário']),
                help=f'Previsto × {pct(inferior)}: só 1 loja em cada 10 deve vender menos')
    col3.metric('Melhor cenário', euros(total['melhor cenário']),
                help=f'Previsto × {pct(superior)}: só 1 loja em cada 10 deve vender mais')
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

    if loja in efeitos.index:
        st.markdown('#### O que pesa na previsão desta loja (valores SHAP)')
        efeito = efeitos.loc[loja].sort_values()
        grafico = go.Figure(go.Bar(x=efeito.values * 100, y=efeito.index, orientation='h',
                                   marker_color=['#C0392B' if v < 0 else '#1F4E79' for v in efeito.values],
                                   hovertemplate='%{y}: %{x:+.1f}%<extra></extra>'))
        grafico.update_layout(height=320, margin=dict(t=10, b=10), xaxis_title='Efeito médio na venda do dia (%)')
        st.plotly_chart(grafico, use_container_width=True)
        st.caption('Quanto cada grupo de variáveis aumenta (azul) ou reduz (vermelho) a venda prevista, em média, nos dias '
                   'abertos das 6 semanas, partindo da venda típica da rede. "Médias da loja" é o tamanho da loja.')

with aba_cenarios:
    st.subheader('Pior, provável e melhor cenário de cada loja nas 6 semanas')
    st.markdown(
        '- **Cenário provável:** o total previsto pelo modelo, para o planejamento normal.\n'
        f'- **Pior cenário:** previsto × **{pct(inferior)}**. É o mínimo com que a loja deve contar (orçamento conservador).\n'
        f'- **Melhor cenário:** previsto × **{pct(superior)}**. Dificilmente a loja vende mais que isso.\n\n'
        'Os percentuais vêm do erro do **total de 6 semanas** medido em 5 períodos passados (inclusive o Natal), e não do '
        'erro de um dia: em 8 de cada 10 lojas, o total real deve ficar entre o pior e o melhor cenário.')
    col1, col2, col3 = st.columns(3)
    col1.metric('Pior cenário da rede', euros(totais['pior cenário'].sum()))
    col2.metric('Cenário provável da rede', euros(totais['cenário provável'].sum()))
    col3.metric('Melhor cenário da rede', euros(totais['melhor cenário'].sum()))
    tabela_cenarios = totais.join(lojas[['tipo_loja', 'sortimento']])
    st.dataframe(tabela_cenarios.style.format(euros, subset=['pior cenário', 'cenário provável', 'melhor cenário']),
                 use_container_width=True, height=380)
    st.download_button('Baixar os cenários (CSV)', tabela_cenarios.round(2).to_csv().encode('utf-8'), 'cenarios_6_semanas.csv')

    st.markdown('#### Os cenários funcionaram no passado? (as 6 semanas do teste, com a venda real já conhecida)')
    teste_cenarios, _ = carregar_monitoramento()
    conferencia = cenarios_por_loja(teste_cenarios['loja'], teste_cenarios['previsto'], (inferior, superior),
                                    real=teste_cenarios['real'])
    onde = conferencia['real caiu'].value_counts(normalize=True)
    col1, col2, col3 = st.columns(3)
    col1.metric('Total real entre os cenários', pct(onde.get('entre os cenários', 0)), help='Esperado: 80%')
    col2.metric('Abaixo do pior cenário', pct(onde.get('abaixo do pior', 0)), help='Esperado: 10%')
    col3.metric('Acima do melhor cenário', pct(onde.get('acima do melhor', 0)), help='Esperado: 10%')
    st.dataframe(conferencia.style.format({'pior cenário': euros, 'cenário provável': euros, 'melhor cenário': euros,
                                           'total real': euros, 'erro do total': lambda v: f'{v:+.1%}',
                                           'MAE do dia': euros, 'MAPE do dia': pct}),
                 use_container_width=True, height=380)
    st.caption('Modelo final treinado até 18/06/2015, sem ver o teste. MAE e MAPE do dia: erro médio da venda de um dia '
               'da loja, só como informação. Os cenários usam o erro do total, porque os erros dos dias se compensam em '
               'parte ao longo das 6 semanas.')

with aba_reforma:
    st.subheader('Quanto reservar para a reforma de cada loja?')
    percentual = st.slider('Parcela do faturamento das 6 semanas destinada à reforma (%)', 1, 20, 5)
    conservador = st.toggle('Usar o pior cenário (orçamento conservador)', value=False)
    base_calculo = totais['pior cenário'] if conservador else totais['cenário provável']
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
    st.markdown(f"**Modelo:** {ficha['modelo']}  \n**Treinado com:** {ficha['treinado_com']}  \n"
                f"**Hiperparâmetros escolhidos por:** {ficha.get('metodo_de_busca', '—')}")
    st.markdown('**Avisos:**\n' + '\n'.join(f'- {aviso}' for aviso in ficha['avisos']))

### Monitoramento: as 6 semanas do teste acompanhadas semana a semana, como se fossem a produção
with aba_monitoramento:
    sla = ficha['monitoramento']
    teste, referencia = carregar_monitoramento()
    st.markdown('Simulação de produção: o modelo final (treinado até 18/06/2015) prevê as **6 semanas do teste**, e a venda '
                'real de cada semana chega depois. O **SLA** vem das 30 semanas da validação cruzada no tempo '
                '(o modelo retreinado a cada 6 semanas, como em produção).')
    numero = ((teste['data'] - teste['data'].min()).dt.days // 7).clip(upper=5) + 1
    inicio = teste.groupby(numero)['data'].transform('min')
    fim = teste.groupby(numero)['data'].transform('max')
    rotulos = 'S' + numero.astype(str) + ': ' + inicio.dt.strftime('%d/%m') + ' a ' + fim.dt.strftime('%d/%m')

    escolha = st.selectbox('Acompanhar', ['Rede inteira'] + [f'Loja {v}' for v in sorted(teste['loja'].unique())])
    filtro = ((teste['loja'] == int(escolha.removeprefix('Loja '))) if escolha != 'Rede inteira'
              else pd.Series(True, index=teste.index))
    painel = painel_semanas(teste.loc[filtro, 'real'], teste.loc[filtro, 'previsto'], rotulos[filtro],
                            sla['sla_mape_semanal'], sla['sla_erro_total_rede_semanal'])
    fora = int((painel['Situação'] == '🚨 fora do SLA').sum())

    col1, col2, col3 = st.columns(3)
    col1.metric('SLA: MAPE da semana', f"≤ {pct(sla['sla_mape_semanal'])}")
    col2.metric('SLA: erro do total da semana', f"≤ ±{pct(sla['sla_erro_total_rede_semanal'])}")
    col3.metric('Semanas fora do SLA', f'{fora} de {len(painel)}')
    st.dataframe(painel.style.format({'Dias de loja': '{:,.0f}', 'MAE (€)': euros, 'MAPE': pct, 'MPE': lambda v: f'{v:+.1%}',
                                      'Erro do total da rede': lambda v: f'{v:+.1%}'}), use_container_width=True)
    st.caption('O SLA foi feito para a rede inteira. Numa loja só, o erro de uma semana oscila muito mais, e '
               '"fora do SLA" serve só como sinal para olhar a loja de perto.')

    st.markdown('#### Data drift: as 6 semanas do teste × as semanas anteriores (não precisa da venda real)')
    drift = relatorio_drift(referencia, teste, sla['variaveis_drift'])
    st.dataframe(drift.style.format({'PSI': '{:.3f}', 'K-S (distância)': '{:.3f}', 'K-S (p-valor)': '{:.3g}'}),
                 use_container_width=True)
    st.caption(f'Referência: amostra de {len(referencia):,} dias de loja das 5 janelas da validação cruzada. '.replace(',', '.') +
               f"PSI até 0,10: estável; até 0,25: atenção; acima de {sla['psi_alerta']:.2f}: mudança. Variáveis de calendário "
               '(dia, mês, semana) ficam de fora: mudam de um período para outro por construção.')
