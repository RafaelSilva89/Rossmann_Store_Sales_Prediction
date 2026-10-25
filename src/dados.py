"""Leitura, tradução, limpeza e criação de variáveis da base Rossmann (vendas diárias + cadastro das lojas)."""
import numpy as np
import pandas as pd

from src.config import COLUNAS, DISTANCIA_SEM_CONCORRENTE, FERIADOS, SORTIMENTOS

MESES_PROMO2 = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sept': 9,
                'Oct': 10, 'Nov': 11, 'Dec': 12}


def carregar_lojas(caminho):
    """Cadastro das 1.115 lojas, com colunas e códigos traduzidos."""
    lojas = pd.read_csv(caminho).rename(columns=COLUNAS)
    lojas['sortimento'] = lojas['sortimento'].map(SORTIMENTOS)
    return lojas


def carregar_base(caminho_vendas, caminho_lojas):
    """Vendas diárias juntadas ao cadastro das lojas, traduzidas e com tipos enxutos (economiza memória)."""
    vendas = pd.read_csv(caminho_vendas, dtype={'StateHoliday': str}, parse_dates=['Date']).rename(columns=COLUNAS)
    vendas['feriado'] = vendas['feriado'].map(FERIADOS)
    df = vendas.merge(carregar_lojas(caminho_lojas), on='loja', how='left')
    for coluna in df.select_dtypes('int64').columns:
        df[coluna] = pd.to_numeric(df[coluna], downcast='integer')
    for coluna in df.select_dtypes('float64').columns:
        df[coluna] = df[coluna].astype('float32')
    for coluna in ['feriado', 'tipo_loja', 'sortimento', 'promo2_meses']:
        df[coluna] = df[coluna].astype('category')
    return df.sort_values(['data', 'loja']).reset_index(drop=True)


def criar_variaveis(df):
    """Variáveis de calendário, concorrência e Promo2 (nenhuma usa as vendas).

    - ano, mes, dia, semana_ano: calendário;
    - sem_concorrente: 1 quando a distância do concorrente não foi informada (vira 100 km, longe de todos);
    - meses_concorrente: meses desde a abertura do concorrente (negativo = ainda não tinha aberto);
      concorrente_data_desconhecida = 1 quando a data não foi informada (e meses_concorrente vira 0);
    - semanas_promo2: semanas desde a adesão à Promo2 (0 para quem não participa ou ainda não aderiu);
    - promo2_mes_ativo: 1 quando a loja participa da Promo2 e o mês está entre os meses de reinício da promoção.
    """
    df = df.copy()
    data = df['data']
    df['ano'] = data.dt.year.astype('int16')
    df['mes'] = data.dt.month.astype('int8')
    df['dia'] = data.dt.day.astype('int8')
    df['semana_ano'] = data.dt.isocalendar().week.astype('int8')

    df['sem_concorrente'] = df['distancia_concorrente'].isna().astype('int8')
    df['distancia_concorrente'] = df['distancia_concorrente'].fillna(DISTANCIA_SEM_CONCORRENTE).astype('float32')
    desconhecida = df['concorrente_desde_ano'].isna()
    meses = (df['ano'] - df['concorrente_desde_ano']) * 12 + (df['mes'] - df['concorrente_desde_mes'])
    df['meses_concorrente'] = meses.where(~desconhecida, 0).astype('float32')
    df['concorrente_data_desconhecida'] = desconhecida.astype('int8')

    inicio_promo2 = pd.to_datetime(df['promo2_desde_ano'].astype('Int64').astype(str) + '-W'
                                   + df['promo2_desde_semana'].astype('Int64').astype(str) + '-1',
                                   format='%G-W%V-%u', errors='coerce')
    semanas = ((data - inicio_promo2).dt.days / 7).clip(lower=0)
    df['semanas_promo2'] = semanas.where(df['promo2'] == 1, 0).fillna(0).astype('float32')
    mes_ativo = np.zeros(len(df), dtype=bool)
    intervalo = df['promo2_meses'].astype('object').fillna('')
    for texto in intervalo.unique():
        if not texto:
            continue
        meses_do_intervalo = [MESES_PROMO2[mes] for mes in texto.split(',')]
        mes_ativo |= ((intervalo == texto) & df['mes'].isin(meses_do_intervalo)).to_numpy()
    df['promo2_mes_ativo'] = (mes_ativo & (df['semanas_promo2'] > 0).to_numpy()).astype('int8')
    return df


def filtrar_dias_com_venda(df):
    """Mantém só os dias em que a loja abriu e vendeu (loja fechada vende 0: não há o que aprender)."""
    return df[(df['aberta'] == 1) & (df['vendas'] > 0)].reset_index(drop=True)


def separar_periodos(df, inicio_validacao, inicio_teste):
    """Separação temporal: treino (antes da validação), validação e teste (as 6 últimas semanas)."""
    treino = df[df['data'] < inicio_validacao]
    validacao = df[(df['data'] >= inicio_validacao) & (df['data'] < inicio_teste)]
    teste = df[df['data'] >= inicio_teste]
    return treino, validacao, teste


def carregar_futuro(caminho_futuro, caminho_lojas):
    """As 6 semanas a prever (test.csv do Kaggle, sem as vendas), no mesmo formato da base de treino.

    11 linhas da loja 622 não informam se a loja abre; como são dias úteis, consideramos aberta.
    """
    futuro = pd.read_csv(caminho_futuro, dtype={'StateHoliday': str}, parse_dates=['Date']).rename(columns=COLUNAS)
    futuro['aberta'] = futuro['aberta'].fillna(1).astype('int8')
    futuro['feriado'] = futuro['feriado'].map(FERIADOS)
    df = futuro.merge(carregar_lojas(caminho_lojas), on='loja', how='left')
    for coluna in ['feriado', 'tipo_loja', 'sortimento', 'promo2_meses']:
        df[coluna] = df[coluna].astype('category')
    return criar_variaveis(df.sort_values(['data', 'loja']).reset_index(drop=True))
