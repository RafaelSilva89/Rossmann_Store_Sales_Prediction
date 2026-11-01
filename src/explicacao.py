"""Explicação das previsões do XGBoost pelos valores SHAP (calculados pelo próprio XGBoost, sem a biblioteca shap)."""
import numpy as np
import pandas as pd
import xgboost as xgb

# Variáveis do modelo reunidas em grupos com sentido de negócio (as 3 médias da loja medem o "tamanho" dela)
GRUPOS_SHAP = {
    'Médias da loja': ['media_loja', 'media_loja_dia_semana', 'media_loja_promo'],
    'Promoção do dia': ['promo'],
    'Dia da semana': ['dia_semana'],
    'Calendário (dia, mês, semana, ano)': ['dia', 'mes', 'semana_ano', 'ano'],
    'Feriados': ['feriado', 'feriado_escolar'],
    'Concorrência': ['distancia_concorrente', 'sem_concorrente', 'meses_concorrente', 'concorrente_data_desconhecida'],
    'Promo2': ['promo2', 'semanas_promo2', 'promo2_mes_ativo'],
    'Tipo e sortimento': ['tipo_loja', 'sortimento'],
}


def contribuicoes_xgboost(modelo, X):
    """Contribuição de cada variável para a previsão de cada dia de loja (valores SHAP, na escala do log da venda).

    `modelo` é o TransformedTargetRegressor do projeto (médias da loja + one-hot + XGBoost, alvo em log1p).
    A coluna 'base' é o ponto de partida (a média do modelo); base + soma das contribuições = log1p(previsão).
    Como a escala é log, cada contribuição MULTIPLICA a venda: exp(contribuição) - 1 é o efeito em %.
    """
    pipeline = modelo.regressor_
    preparado = pipeline.named_steps['preparo'].transform(pipeline.named_steps['medias'].transform(X))
    nomes = list(pipeline.named_steps['preparo'].get_feature_names_out())
    booster = pipeline.named_steps['modelo'].get_booster()
    valores = booster.predict(xgb.DMatrix(np.asarray(preparado, dtype=float), feature_names=nomes), pred_contribs=True)
    return pd.DataFrame(valores, index=X.index, columns=nomes + ['base'])


def agrupar_contribuicoes(contribuicoes, grupos=GRUPOS_SHAP):
    """Soma as contribuições por grupo de negócio (as colunas do one-hot, como 'tipo_loja_a', voltam ao grupo)."""
    def grupo_da(coluna):
        exato = [g for g, variaveis in grupos.items() if coluna in variaveis]   # 'dia_semana' não é one-hot de 'dia'
        prefixo = [g for g, variaveis in grupos.items() if any(coluna.startswith(v + '_') for v in variaveis)]
        return (exato or prefixo or [None])[0]

    por_grupo = pd.Series({coluna: grupo_da(coluna) for coluna in contribuicoes.columns if coluna != 'base'})
    return pd.DataFrame({grupo: contribuicoes[por_grupo.index[por_grupo == grupo]].sum(axis=1) for grupo in grupos},
                        index=contribuicoes.index)
