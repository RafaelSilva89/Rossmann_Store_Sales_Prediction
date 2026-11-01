"""Testes das funções de src/: leitura, variáveis criadas, médias da loja, dobras temporais, métricas,
busca de hiperparâmetros, SHAP e monitoramento."""
import numpy as np
import pandas as pd
import pytest

from src.avaliacao import (cenarios_por_loja, cobertura_faixa, erro_total_6_semanas, faixa_relativa, ml_error,
                           totais_por_loja)
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


def test_cenarios_por_loja():
    futuro = cenarios_por_loja([1, 1, 2], [100, 100, 50], (0.9, 1.1))
    assert list(futuro.columns) == ['pior cenário', 'cenário provável', 'melhor cenário']
    assert futuro.loc[1].tolist() == pytest.approx([180, 200, 220])
    teste = cenarios_por_loja([1, 1, 2, 2], [100, 100, 50, 50], (0.9, 1.1), real=[100, 80, 40, 40])
    assert teste.loc[1, 'MAE do dia'] == pytest.approx(10) and teste.loc[1, 'MAPE do dia'] == pytest.approx(0.125)
    assert teste.loc[1, 'erro do total'] == pytest.approx(200 / 180 - 1)
    assert teste.loc[1, 'real caiu'] == 'entre os cenários' and teste.loc[2, 'real caiu'] == 'abaixo do pior'


def test_testes_estatisticos():
    assert teste_mann_whitney([5, 6, 7], [1, 2, 3])['tamanho do efeito'] == pytest.approx(1)
    assert teste_kruskal([1, 2, 3, 10, 11, 12], list('aaabbb'))['tamanho do efeito'] > 0.5


def modelo_pequeno():
    """Pipeline no formato do projeto (médias da loja + one-hot + XGBoost, alvo em log), com dados sintéticos."""
    import xgboost as xgb
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    rng = np.random.default_rng(0)
    X = pd.DataFrame({'loja': rng.integers(1, 6, 600), 'dia_semana': rng.integers(1, 7, 600), 'promo': rng.integers(0, 2, 600),
                      'tipo_loja': rng.choice(['a', 'b'], 600), 'dia': rng.integers(1, 29, 600)})
    y = pd.Series(1000 * X['loja'] * (1 + 0.3 * X['promo']) + rng.normal(0, 100, 600))
    preparo = ColumnTransformer([('categorias', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['tipo_loja'])],
                                remainder='passthrough', verbose_feature_names_out=False)
    pipeline = Pipeline([('medias', MediasDaLoja()), ('preparo', preparo),
                         ('modelo', xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=0))])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1), X, y


def test_shap_soma_o_log_da_previsao_e_agrupa():
    from src.explicacao import agrupar_contribuicoes, contribuicoes_xgboost

    modelo, X, y = modelo_pequeno()
    modelo.fit(X, y)
    contrib = contribuicoes_xgboost(modelo, X.head(20))
    assert np.allclose(contrib.sum(axis=1), np.log1p(modelo.predict(X.head(20))), atol=1e-3)
    grupos = agrupar_contribuicoes(contrib)
    assert np.allclose(grupos.sum(axis=1) + contrib['base'], contrib.sum(axis=1), atol=1e-5)   # nenhuma coluna se perde
    assert np.allclose(grupos['Dia da semana'], contrib['dia_semana'])   # 'dia_semana' não cai no grupo de 'dia'
    assert np.allclose(grupos['Tipo e sortimento'], contrib[['tipo_loja_a', 'tipo_loja_b']].sum(axis=1))


def test_optuna_com_janelas_minimiza_e_nao_poda_na_ultima():
    import optuna
    from sklearn.metrics import mean_absolute_percentage_error
    from src.otimizacao import modelos_treinados_optuna, objetivo_cv_com_poda, resumo_optuna

    class PodaSempre(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return True

    modelo, X, y = modelo_pequeno()
    janelas = [(np.arange(0, 300), np.arange(300, 400)), (np.arange(0, 400), np.arange(400, 500))]
    espaco = lambda t: {'regressor__modelo__max_depth': t.suggest_int('max_depth', 2, 4)}
    objetivo = objetivo_cv_com_poda(modelo, espaco, X, y, janelas, mean_absolute_percentage_error)
    estudo = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0))
    estudo.optimize(objetivo, n_trials=2)
    linha = resumo_optuna('Optuna', estudo, 1.0, n_splits=2)
    assert 0 < linha['Melhor score (CV)'] < 0.5 and linha['Modelos treinados'] == 4
    assert set(linha['Melhores hiperparâmetros']) == {'max_depth'}

    podando = optuna.create_study(direction='minimize', pruner=PodaSempre())
    podando.optimize(objetivo, n_trials=1)
    assert podando.trials[0].state == optuna.trial.TrialState.PRUNED and modelos_treinados_optuna(podando, 2) == 1
    uma_janela = optuna.create_study(direction='minimize', pruner=PodaSempre())
    uma_janela.optimize(objetivo_cv_com_poda(modelo, espaco, X, y, janelas[:1], mean_absolute_percentage_error), n_trials=1)
    assert uma_janela.trials[0].state == optuna.trial.TrialState.COMPLETE   # na última janela não há poda


def test_historico_minimizando_ignora_as_podadas():
    import optuna
    from src.otimizacao import historico_optuna

    estudo = optuna.create_study(direction='minimize')
    estudo.add_trial(optuna.trial.create_trial(value=0.10, intermediate_values={0: 0.10, 1: 0.10}))
    estudo.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED, value=0.05, intermediate_values={0: 0.05}))
    estudo.add_trial(optuna.trial.create_trial(value=0.09, intermediate_values={0: 0.09, 1: 0.09}))
    valores, melhor = historico_optuna(estudo, maior_melhor=False)
    assert np.isnan(valores[1]) and list(melhor) == [0.10, 0.10, 0.09]


def test_psi_funciona_com_variavel_binaria_e_drift():
    from src.monitoramento import psi, relatorio_drift

    rng = np.random.default_rng(1)
    assert psi(rng.integers(0, 2, 5000), rng.integers(0, 2, 5000)) < 0.01
    assert psi(rng.random(5000) < 0.4, rng.random(5000) < 0.9) > 0.25   # promoção em 40% x 90% dos dias
    referencia = pd.DataFrame({'igual': rng.normal(size=5000), 'deslocada': rng.normal(size=5000)})
    atual = pd.DataFrame({'igual': rng.normal(size=5000), 'deslocada': rng.normal(1, 1, size=5000)})
    drift = relatorio_drift(referencia, atual, ['igual', 'deslocada'])
    assert list(drift['Leitura']) == ['Estável', 'Mudança']


def test_painel_semanas_marca_a_semana_fora_do_sla():
    from src.monitoramento import painel_semanas

    real = np.array([100, 100, 100, 100, 100, 100.0])
    previsto = np.array([105, 95, 100, 130, 125, 120.0])
    painel = painel_semanas(real, previsto, ['s1'] * 3 + ['s2'] * 3, sla_mape=0.10, sla_total_rede=0.05)
    assert painel.loc['s1', 'MAPE'] == pytest.approx(10 / 300) and painel.loc['s2', 'Erro do total da rede'] == pytest.approx(0.25)
    assert list(painel['Situação']) == ['✅ dentro do SLA', '🚨 fora do SLA']
