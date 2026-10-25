"""Seleção de variáveis por dois métodos: importância do Random Forest e Informação Mútua."""
from functools import partial

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression


def importancia_random_forest(X, y, tipo='regressao', semente=42, **parametros_floresta):
    """Importância de cada coluna para um Random Forest treinado em X (quanto ela ajudou a dividir as árvores).

    parametros_floresta: repassados ao Random Forest (ex.: menos árvores ou folhas maiores, para caber na memória).
    """
    floresta = RandomForestRegressor if tipo == 'regressao' else RandomForestClassifier
    modelo = floresta(**{'random_state': semente, 'n_jobs': 2, **parametros_floresta}).fit(X, y)
    return pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)


def selecao_features(X, y, k, tipo='regressao', semente=42):
    """Seleção das k melhores colunas por Informação Mútua (SelectKBest).

    Devolve o ranking de todas as colunas (feature, score), do maior para o menor, e o seletor ajustado.
    O random_state fixo faz o resultado ser o mesmo a cada execução.
    """
    funcao = mutual_info_regression if tipo == 'regressao' else mutual_info_classif
    seletor = SelectKBest(score_func=partial(funcao, random_state=semente), k=k).fit(X, y)
    df_scores = (pd.DataFrame({'feature': X.columns, 'score': seletor.scores_})
                 .sort_values('score', ascending=False).reset_index(drop=True))
    return df_scores, seletor
