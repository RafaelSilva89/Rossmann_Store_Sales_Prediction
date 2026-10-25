"""Testes das funções de src/: leitura, variáveis criadas, médias da loja, dobras temporais e métricas."""
import numpy as np
import pandas as pd
import pytest

from src.avaliacao import cobertura_faixa, erro_total_6_semanas, faixa_relativa, ml_error, totais_por_loja
from src.config import ARQUIVO_LOJAS, DISTANCIA_SEM_CONCORRENTE, FEATURES, PROIBIDAS
from src.dados import carregar_lojas, criar_variaveis, separar_periodos
from src.estatistica import teste_kruskal, teste_mann_whitney
from src.modelos import MediaPorGrupo, MediasDaLoja
from src.validacao import dobras_temporais


def dia_de_loja(**campos):
    """Uma linha no formato da base juntada (vendas + cadastro), com valores padrão."""
    linha = dict(loja=1, data=pd.Timestamp('2015-03-10'), dia_semana=2, promo=1, feriado='nenhum', feriado_escolar=0,
                 tipo_loja='a', sortimento='básico', distancia_concorrente=500.0, concorrente_desde_mes=1.0,
                 concorrente_desde_ano=2014.0, promo2=0, promo2_desde_semana=np.nan, promo2_desde_ano=np.nan,
                 promo2_meses=np.nan)
    linha.update(campos)
    return pd.DataFrame([linha])


def test_carregar_lojas_traduz():
    lojas = carregar_lojas(ARQUIVO_LOJAS)
    assert len(lojas) == 1115 and lojas['loja'].is_unique
    assert set(lojas['sortimento']) == {'básico', 'extra', 'estendido'}


def test_variaveis_de_concorrencia():
    criada = criar_variaveis(dia_de_loja()).iloc[0]
    assert criada['meses_concorrente'] == 14 and criada['concorrente_data_desconhecida'] == 0
    sem = criar_variaveis(dia_de_loja(distancia_concorrente=np.nan, concorrente_desde_mes=np.nan,
                                      concorrente_desde_ano=np.nan)).iloc[0]
    assert sem['sem_concorrente'] == 1 and sem['distancia_concorrente'] == DISTANCIA_SEM_CONCORRENTE
    assert sem['concorrente_data_desconhecida'] == 1 and sem['meses_concorrente'] == 0


def test_variaveis_de_promo2():
    # Aderiu na semana ISO 1 de 2015 (começa em 29/12/2014); reinicia em jan, abr, jul e out
    adesao = dict(promo2=1, promo2_desde_semana=1.0, promo2_desde_ano=2015.0, promo2_meses='Jan,Apr,Jul,Oct')
    abril = criar_variaveis(dia_de_loja(data=pd.Timestamp('2015-04-06'), **adesao)).iloc[0]
    assert abril['semanas_promo2'] == pytest.approx(14) and abril['promo2_mes_ativo'] == 1
    marco = criar_variaveis(dia_de_loja(data=pd.Timestamp('2015-03-02'), **adesao)).iloc[0]
    assert marco['promo2_mes_ativo'] == 0
    antes = criar_variaveis(dia_de_loja(data=pd.Timestamp('2014-10-06'), **adesao)).iloc[0]
    assert antes['semanas_promo2'] == 0 and antes['promo2_mes_ativo'] == 0   # ainda não tinha aderido


def test_clientes_fora_do_modelo():
    assert not set(PROIBIDAS) & set(FEATURES)


def test_medias_da_loja_usam_so_o_treino():
    X = pd.DataFrame({'loja': [1, 1, 2, 2], 'dia_semana': [1, 2, 1, 2], 'promo': [0, 1, 0, 1]})
    medias = MediasDaLoja().fit(X, [10, 20, 100, 200])
    novo = medias.transform(pd.DataFrame({'loja': [1, 3], 'dia_semana': [2, 1], 'promo': [1, 0]}))
    assert 'loja' not in novo.columns
    assert novo['media_loja'].tolist() == [15, 82.5]   # loja 3 nunca vista: média de todas as lojas
    assert novo['media_loja_dia_semana'].tolist() == [20, 82.5]


def test_media_por_grupo():
    X = pd.DataFrame({'loja': [1, 1, 2]})
    modelo = MediaPorGrupo(('loja',)).fit(X, [10, 30, 50])
    assert modelo.predict(pd.DataFrame({'loja': [1, 2, 9]})).tolist() == [20, 50, 30]


def test_dobras_temporais_nao_olham_o_futuro():
    datas = pd.Series(pd.date_range('2015-01-01', '2015-06-30'))
    dobras = dobras_temporais(datas, n_dobras=3, semanas=6)
    assert len(dobras) == 3
    for treino, avaliacao in dobras:
        assert datas[treino].max() < datas[avaliacao].min()
        assert len(avaliacao) == 42
    assert datas[dobras[-1][1]].max() == datas.max()


def test_separar_periodos():
    df = pd.DataFrame({'data': pd.date_range('2015-05-01', '2015-07-31')})
    treino, validacao, teste = separar_periodos(df, '2015-05-07', '2015-06-19')
    assert len(treino) + len(validacao) + len(teste) == len(df)
    assert teste['data'].min() == pd.Timestamp('2015-06-19') and len(teste) == 43


def test_metricas_e_totais():
    metricas = ml_error('m', [100, 200], [110, 180]).iloc[0]
    assert metricas['MAE (€)'] == pytest.approx(15) and metricas['MAPE'] == pytest.approx(0.1)
    assert metricas['MPE'] == pytest.approx(0)
    totais = totais_por_loja([1, 1, 2], [100, 100, 50], [90, 90, 60])
    assert totais.loc[1, 'erro relativo'] == pytest.approx(-0.1)
    assert erro_total_6_semanas([1, 1, 2], [100, 100, 50], [90, 90, 60]) == pytest.approx(0.15)


def test_faixa_relativa_e_cobertura():
    real = np.array([90, 100, 110, 120])
    faixa = faixa_relativa(real, np.full(4, 100.0), cobertura=0.5)
    assert faixa[0] <= 1 <= faixa[1]
    assert cobertura_faixa([95, 150], [100, 100], (0.9, 1.1)) == 0.5


def test_testes_estatisticos():
    assert teste_mann_whitney([5, 6, 7], [1, 2, 3])['tamanho do efeito'] == pytest.approx(1)
    assert teste_kruskal([1, 2, 3, 10, 11, 12], list('aaabbb'))['tamanho do efeito'] > 0.5
