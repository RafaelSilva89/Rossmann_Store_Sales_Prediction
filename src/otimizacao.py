"""Comparação de métodos de ajuste de hiperparâmetros: Grid Search, Random Search e Bayesiana (Optuna, com poda)."""
import numpy as np
import optuna
from sklearn.base import clone


def _nome(parametro):
    """'regressor__modelo__max_depth' -> 'max_depth' (o nome do hiperparâmetro sem o caminho do Pipeline)."""
    return parametro.split('__')[-1]


def objetivo_cv_com_poda(modelo, espaco, X, y, dobras, metrica):
    """Cria a função objetivo do Optuna com poda (*pruning*): a validação cruzada roda janela a janela.

    `dobras` é a lista de (índices de treino, índices de avaliação), como a de `dobras_temporais`; `espaco`
    recebe o trial e devolve o dicionário de hiperparâmetros (nomes do Pipeline); `metrica(y_real, previsto)`
    é o erro a minimizar (ex.: MAPE). Depois de cada janela, o erro médio até ali é informado ao Optuna
    (`trial.report`); se estiver acima da mediana das tentativas anteriores na mesma janela, o pruner interrompe
    a tentativa e as janelas restantes não são treinadas. Na última janela não há poda: o custo já foi pago.
    O desvio entre as janelas e o erro no treino ficam guardados no trial, para a comparação com o sklearn.
    """
    dobras = list(dobras)

    def objetivo(trial):
        estimador = clone(modelo).set_params(**espaco(trial))
        teste, treino = [], []
        for passo, (i_treino, i_teste) in enumerate(dobras):
            ajustado = clone(estimador).fit(X.iloc[i_treino], y.iloc[i_treino])
            teste.append(metrica(y.iloc[i_teste], ajustado.predict(X.iloc[i_teste])))
            treino.append(metrica(y.iloc[i_treino], ajustado.predict(X.iloc[i_treino])))
            trial.report(float(np.mean(teste)), passo)
            if passo < len(dobras) - 1 and trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr('desvio', float(np.std(teste)))
        trial.set_user_attr('treino', float(np.mean(treino)))
        return float(np.mean(teste))
    return objetivo


def modelos_treinados_optuna(study, n_splits):
    """Quantos modelos o estudo treinou: as tentativas podadas param no meio da validação cruzada."""
    podada = optuna.trial.TrialState.PRUNED
    return int(sum(len(t.intermediate_values) if t.state == podada else n_splits for t in study.trials))


def resumo_sklearn(nome, busca, tempo, sinal=1):
    """Linha da tabela de comparação a partir de um GridSearchCV/RandomizedSearchCV (com return_train_score=True).

    `sinal=-1` desfaz o sinal negativo das métricas de erro do sklearn (ex.: neg_mean_absolute_percentage_error).
    """
    i = busca.best_index_
    r = busca.cv_results_
    return {'Método': nome, 'Melhor score (CV)': sinal * r['mean_test_score'][i], 'Desvio entre dobras': r['std_test_score'][i],
            'Score no treino': sinal * r['mean_train_score'][i], 'Combinações testadas': len(r['params']),
            'Modelos treinados': len(r['params']) * busca.n_splits_, 'Tempo (s)': tempo,
            'Melhores hiperparâmetros': {_nome(k): v for k, v in busca.best_params_.items()}}


def resumo_optuna(nome, study, tempo, n_splits):
    """Linha da tabela de comparação a partir de um estudo do Optuna criado com `objetivo_cv_com_poda`."""
    melhor = study.best_trial
    return {'Método': nome, 'Melhor score (CV)': melhor.value, 'Desvio entre dobras': melhor.user_attrs['desvio'],
            'Score no treino': melhor.user_attrs['treino'], 'Combinações testadas': len(study.trials),
            'Modelos treinados': modelos_treinados_optuna(study, n_splits), 'Tempo (s)': tempo,
            'Melhores hiperparâmetros': {_nome(k): v for k, v in melhor.params.items()}}


def historico_optuna(study, maior_melhor=True):
    """Score de cada trial (NaN nas podadas) e o melhor score acumulado até ele (para o gráfico do histórico).

    No Optuna 5, a tentativa podada guarda como `value` o erro médio até a poda: por isso o filtro é pelo estado.
    """
    completa = optuna.trial.TrialState.COMPLETE
    valores = np.array([t.value if t.state == completa else np.nan for t in study.trials], dtype=float)
    acumulado = np.fmax.accumulate(valores) if maior_melhor else np.fmin.accumulate(valores)
    return valores, acumulado
