"""Validação cruzada temporal: o modelo sempre treina no passado e é avaliado nas 6 semanas seguintes."""
import numpy as np
import pandas as pd


def dobras_temporais(datas, n_dobras=5, semanas=6):
    """Dobras de janela crescente, do passado para o presente, no formato que o scikit-learn aceita em `cv`.

    A última dobra avalia as `semanas` finais de `datas`; cada dobra anterior recua mais `semanas`.
    O treino de cada dobra é tudo o que vem antes da sua janela de avaliação (nunca o futuro).
    """
    datas = pd.Series(pd.to_datetime(datas)).reset_index(drop=True)
    fim = datas.max()
    dobras = []
    for k in reversed(range(1, n_dobras + 1)):
        inicio_avaliacao = fim - pd.Timedelta(days=k * semanas * 7 - 1)
        fim_avaliacao = fim - pd.Timedelta(days=(k - 1) * semanas * 7)
        treino = np.flatnonzero(datas < inicio_avaliacao)
        avaliacao = np.flatnonzero((datas >= inicio_avaliacao) & (datas <= fim_avaliacao))
        dobras.append((treino, avaliacao))
    return dobras


def periodos_das_dobras(datas, dobras):
    """Tabela com o período de treino e de avaliação de cada dobra (para mostrar no notebook)."""
    datas = pd.Series(pd.to_datetime(datas)).reset_index(drop=True)
    return pd.DataFrame([{'dobra': i, 'treino até': datas[treino].max().date(),
                          'avaliação de': datas[avaliacao].min().date(), 'avaliação até': datas[avaliacao].max().date(),
                          'linhas de treino': len(treino), 'linhas de avaliação': len(avaliacao)}
                         for i, (treino, avaliacao) in enumerate(dobras, 1)]).set_index('dobra')
