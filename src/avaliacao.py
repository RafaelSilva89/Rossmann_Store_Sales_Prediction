"""Avaliação dos modelos de regressão: métricas (padrão ml_error), relatório, comparação, totais por loja e ficha."""
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.graficos import COR_DESTAQUE, COR_PRINCIPAL, euros, salvar, titulo


def ml_error(nome, y, yhat):
    """Métricas de regressão de um modelo, com os erros em euros (€) da venda diária.

    MAE: erro absoluto médio · MAPE: erro percentual médio (em relação à venda real) ·
    RMSE: raiz do erro quadrático médio (pune mais os erros grandes) · R²: parte das diferenças explicada ·
    MPE: erro percentual médio COM sinal (positivo = o modelo superestima).
    """
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return pd.DataFrame({
        'MAE (€)': mean_absolute_error(y, yhat),
        'MAPE': np.mean(np.abs((y - yhat) / y)),
        'RMSE (€)': root_mean_squared_error(y, yhat),
        'R²': r2_score(y, yhat),
        'MPE': np.mean((yhat - y) / y),
    }, index=[nome])


def totais_por_loja(lojas, y, yhat):
    """Soma da venda real e da prevista de cada loja no período (o número que o CFO usa) e o erro relativo."""
    tabela = pd.DataFrame({'loja': np.asarray(lojas), 'real': np.asarray(y, dtype=float),
                           'previsto': np.asarray(yhat, dtype=float)}).groupby('loja')[['real', 'previsto']].sum()
    tabela['erro relativo'] = (tabela['previsto'] - tabela['real']) / tabela['real']
    return tabela


def erro_total_6_semanas(lojas, y, yhat):
    """MAPE do total do período por loja: em média, quanto o total previsto de uma loja erra o total real."""
    return float(totais_por_loja(lojas, y, yhat)['erro relativo'].abs().mean())


def relatorio_validacao(nome, modelo, X_treino, y_treino, X_validacao, y_validacao, nome_arquivo=None):
    """Avalia um modelo já treinado na validação, no formato padrão do projeto.

    Mostra: amostra gabarito x previsão, métricas (ml_error), erro do total de 6 semanas por loja, MAE no
    treino x validação (sobreajuste) e os gráficos de previsto x real e de distribuição do erro.
    Devolve (métricas, previsões da validação).
    """
    previsto = modelo.predict(X_validacao)
    real = np.asarray(y_validacao, dtype=float)
    print('AMOSTRA: GABARITO x PREVISÃO DA MÁQUINA (venda do dia em €)')
    display(pd.DataFrame({'Loja': X_validacao['loja'].to_numpy(), 'Gabarito': real, 'Previsão da máquina': previsto,
                          'Erro (€)': previsto - real}).sample(6, random_state=0).round(0))

    metricas = ml_error(nome, y_validacao, previsto)
    metricas['Erro do total por loja'] = erro_total_6_semanas(X_validacao['loja'], real, previsto)
    mae_treino = mean_absolute_error(y_treino, modelo.predict(X_treino))
    metricas['MAE treino (€)'] = mae_treino
    print('=' * 55)
    print(f'MÉTRICAS DE VALIDAÇÃO: {nome}')
    print('=' * 55)
    print(metricas.drop(columns='MAE treino (€)').round(3).T.to_string(header=False))
    print(f'\nMAE no treino: {euros(mae_treino)} | na validação: {euros(metricas["MAE (€)"].iloc[0])} '
          f'| diferença: {metricas["MAE (€)"].iloc[0] - mae_treino:+,.0f} €'.replace(',', '.'))

    amostra = np.random.default_rng(0).choice(len(real), size=min(20_000, len(real)), replace=False)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    axs[0].scatter(previsto[amostra], real[amostra], s=4, alpha=0.2, color=COR_PRINCIPAL)
    limites = [0, max(previsto.max(), real.max())]
    axs[0].plot(limites, limites, color=COR_DESTAQUE, ls='--', label='Previsão perfeita')
    axs[0].set(xlabel='Venda prevista (€)', ylabel='Venda real (€)', title='Previsto x real (amostra de 20 mil dias)')
    axs[0].legend(frameon=False)
    sns.histplot(previsto - real, bins=80, color=COR_PRINCIPAL, ax=axs[1])
    axs[1].axvline(0, color=COR_DESTAQUE, ls='--')
    axs[1].set(xlabel='Erro (previsto - real, €)', ylabel='Dias de loja', title='Distribuição do erro', xlim=(-8000, 8000))
    sns.despine()
    plt.suptitle(nome, fontsize=14, x=0.01, ha='left', y=1.02)
    if nome_arquivo:
        salvar(nome_arquivo)
    plt.show()
    return metricas, previsto


def plot_comparacao_modelos(comparacao, nome_arquivo, referencia='Média da loja (referência)',
                            metricas=('MAE (€)', 'MAPE', 'RMSE (€)', 'Erro do total por loja'), titulo_grafico=None):
    """Painel com uma barra por modelo em cada métrica e a linha do modelo de referência (média de cada loja)."""
    modelos = comparacao.index.tolist()
    cores = [COR_DESTAQUE if m == referencia else COR_PRINCIPAL for m in modelos]
    fig, axes = plt.subplots(1, len(metricas), figsize=(6 * len(metricas), 5.5))
    plt.subplots_adjust(wspace=0.9)
    for ax, metrica in zip(axes, metricas):
        valores = comparacao[metrica]
        ax.barh(modelos, valores, color=cores, alpha=0.85)
        ax.set_xlim(min(0, valores.min()), valores.max() * 1.4)
        for i, v in enumerate(valores):
            texto = f' {v:.3f}' if metrica == 'R²' else f' {v:.1%}' if '€' not in metrica else f' {euros(v)}'
            ax.text(v, i, texto.replace('.', ',') if '€' not in metrica else texto, va='center', fontsize=9)
        if referencia in comparacao.index:
            ax.axvline(comparacao.loc[referencia, metrica], color='gray', ls='--', lw=1, label='média de cada loja')
        ax.set_title(metrica + (' (maior é melhor)' if metrica == 'R²' else ' (menor é melhor)'), fontweight='bold')
        ax.invert_yaxis()
        sns.despine(ax=ax)
    if referencia in comparacao.index:
        fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc='lower center', fontsize=10,
                   bbox_to_anchor=(0.5, -0.06))
    titulo(titulo_grafico or 'Comparação dos modelos na validação')
    salvar(nome_arquivo)
    plt.show()


def faixa_relativa(y, previsto, cobertura=0.80):
    """Faixa de erro RELATIVA: entre quais múltiplos da previsão o valor real costuma ficar.

    Ex.: (0,93; 1,06) = em 80% dos casos, o total real fica entre 93% e 106% do previsto. Uma faixa em
    proporção acompanha o tamanho da loja (um erro de € 10 mil é grande para uma loja pequena e pequeno para uma grande).
    """
    razao = np.asarray(y, dtype=float) / np.asarray(previsto, dtype=float)
    cauda = (1 - cobertura) / 2
    return tuple(float(v) for v in np.quantile(razao, [cauda, 1 - cauda]))


def cobertura_faixa(y, previsto, faixa):
    """Parcela dos casos cujo valor real cai dentro da faixa (previsto x limite inferior, previsto x superior)."""
    y, previsto = np.asarray(y, dtype=float), np.asarray(previsto, dtype=float)
    return float(((y >= previsto * faixa[0]) & (y <= previsto * faixa[1])).mean())


def cenarios_por_loja(lojas, previsto, faixa, real=None):
    """Cenários do total do período de cada loja: pior (previsto x limite inferior da faixa), provável (o previsto)
    e melhor (previsto x limite superior).

    Com a venda real (período já conhecido, como o teste), acrescenta o erro do total, o erro médio do dia (MAE e MAPE)
    e em qual cenário o total real caiu.
    """
    dados = pd.DataFrame({'loja': np.asarray(lojas), 'previsto': np.asarray(previsto, dtype=float)})
    if real is not None:
        dados['real'] = np.asarray(real, dtype=float)
    tabela = dados.groupby('loja')[['previsto']].sum().rename(columns={'previsto': 'cenário provável'})
    tabela.insert(0, 'pior cenário', tabela['cenário provável'] * faixa[0])
    tabela['melhor cenário'] = tabela['cenário provável'] * faixa[1]
    if real is None:
        return tabela
    dados['erro_abs'] = (dados['previsto'] - dados['real']).abs()
    dados['ape'] = dados['erro_abs'] / dados['real']
    por_loja = dados.groupby('loja')
    tabela['total real'] = por_loja['real'].sum()
    tabela['erro do total'] = tabela['cenário provável'] / tabela['total real'] - 1
    tabela['MAE do dia'] = por_loja['erro_abs'].mean()
    tabela['MAPE do dia'] = por_loja['ape'].mean()
    tabela['real caiu'] = np.select([tabela['total real'] < tabela['pior cenário'],
                                     tabela['total real'] > tabela['melhor cenário']],
                                    ['abaixo do pior', 'acima do melhor'], 'entre os cenários')
    return tabela


def erro_por_grupo(y, previsto, grupos):
    """Tradução do erro para o negócio: erro médio por grupo (ex.: tipo de loja) e cenários de melhor e pior caso."""
    tabela = pd.DataFrame({'real': np.asarray(y, dtype=float), 'previsto': np.asarray(previsto, dtype=float),
                           'grupo': pd.Series(grupos).reset_index(drop=True)})
    tabela['erro_abs'] = (tabela['previsto'] - tabela['real']).abs()
    resumo = tabela.groupby('grupo', observed=True).agg(casos=('real', 'size'), venda_real=('real', 'mean'),
                                                        venda_prevista=('previsto', 'mean'), MAE=('erro_abs', 'mean'))
    resumo['MAPE'] = (tabela.assign(ape=tabela['erro_abs'] / tabela['real'])
                      .groupby('grupo', observed=True)['ape'].mean())
    resumo['pior cenário'] = resumo['venda_prevista'] - resumo['MAE']
    resumo['melhor cenário'] = resumo['venda_prevista'] + resumo['MAE']
    return resumo.sort_values('MAPE', ascending=False)


def salvar_ficha_modelo(caminho, **campos):
    """Grava a ficha do modelo (JSON): tudo o que é preciso saber para usá-lo corretamente."""
    ficha = {'data_treino': datetime.now().strftime('%Y-%m-%d %H:%M'), **campos}
    with open(caminho, 'w', encoding='utf-8') as arquivo:
        json.dump(ficha, arquivo, ensure_ascii=False, indent=2, default=float)
    return ficha
