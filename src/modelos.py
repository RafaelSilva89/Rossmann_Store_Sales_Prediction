"""Peças do Pipeline que usam o histórico de vendas da loja, sempre aprendidas só com o período de treino."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

GRUPOS_MEDIAS = {'media_loja': ['loja'], 'media_loja_dia_semana': ['loja', 'dia_semana'],
                 'media_loja_promo': ['loja', 'promo']}


def _media_por_grupo(X, y, grupos):
    return pd.Series(np.asarray(y, dtype=float), index=X.index).groupby([X[g] for g in grupos]).mean()


def _aplicar_media(X, medias, grupos, reserva):
    chave = pd.MultiIndex.from_frame(X[grupos]) if len(grupos) > 1 else X[grupos[0]]
    return pd.Series(medias.reindex(chave).to_numpy(), index=X.index).fillna(reserva)


class MediasDaLoja(BaseEstimator, TransformerMixin):
    """Troca o número da loja pelas vendas médias dela: no geral, por dia da semana e com/sem promoção.

    As médias são calculadas no fit (só com o treino) e aplicadas no transform. Combinação que não apareceu
    no treino recebe a média geral da loja; loja nova recebe a média de todas as lojas.
    """

    def fit(self, X, y):
        self.medias_ = {nome: _media_por_grupo(X, y, grupos) for nome, grupos in GRUPOS_MEDIAS.items()}
        self.media_geral_ = float(np.mean(y))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        media_loja = _aplicar_media(X, self.medias_['media_loja'], ['loja'], self.media_geral_)
        for nome, grupos in GRUPOS_MEDIAS.items():
            X[nome] = media_loja if nome == 'media_loja' else _aplicar_media(X, self.medias_[nome], grupos, media_loja)
        return X.drop(columns='loja')

    def get_feature_names_out(self, nomes_entrada=None):
        nomes = self.feature_names_in_ if nomes_entrada is None else nomes_entrada
        return np.array([nome for nome in nomes if nome != 'loja'] + list(GRUPOS_MEDIAS), dtype=object)


class MediaPorGrupo(BaseEstimator, RegressorMixin):
    """Modelo de referência: prevê a venda média do grupo no treino (ex.: da loja, ou da loja x dia x promoção).

    É o que uma planilha faria. Grupo que não apareceu no treino recebe a média geral.
    """

    def __init__(self, grupos=('loja',)):
        self.grupos = grupos

    def fit(self, X, y):
        self.medias_ = _media_por_grupo(X, y, list(self.grupos))
        self.media_geral_ = float(np.mean(y))
        return self

    def predict(self, X):
        return _aplicar_media(X, self.medias_, list(self.grupos), self.media_geral_).to_numpy()
