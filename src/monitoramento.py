"""Monitoramento do modelo em produção: data drift (PSI e teste K-S) e erro de cada semana contra um SLA."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Faixas usuais do PSI (indústria financeira)
PSI_ATENCAO = 0.10
PSI_MUDANCA = 0.25


def psi(esperado, atual, faixas=10):
    """Population Stability Index: o quanto a distribuição `atual` se afastou da `esperado` (referência).

    As faixas vêm dos percentis da referência. Variáveis com poucos valores (ex.: promoção 0/1) geram percentis
    repetidos, que são unidos; com menos de 2 faixas, cada valor vira uma faixa.
    """
    esperado, atual = np.asarray(esperado, dtype=float), np.asarray(atual, dtype=float)
    limites = np.unique(np.quantile(esperado, np.linspace(0, 1, faixas + 1)))
    if len(limites) < 3:   # variável binária ou quase constante
        limites = np.unique(np.concatenate([limites, [limites[0] - 0.5, limites[-1] + 0.5]]))
    limites[0], limites[-1] = -np.inf, np.inf
    p_esperado = np.histogram(esperado, limites)[0] / len(esperado)
    p_atual = np.histogram(atual, limites)[0] / len(atual)
    p_esperado, p_atual = np.clip(p_esperado, 1e-4, None), np.clip(p_atual, 1e-4, None)   # evita log(0)
    return float(np.sum((p_atual - p_esperado) * np.log(p_atual / p_esperado)))


def leitura_psi(valor):
    """Estável (até 0,10), Atenção (até 0,25) ou Mudança (acima: investigar ou retreinar)."""
    if valor <= PSI_ATENCAO:
        return 'Estável'
    return 'Atenção' if valor <= PSI_MUDANCA else 'Mudança'


def relatorio_drift(referencia, atual, colunas):
    """PSI e teste K-S de cada coluna, comparando os dados `atual` com a `referencia` (DataFrames ou dicionários).

    Não precisa do gabarito. Com muitos dias de loja, o p-valor do K-S fica perto de 0 até para diferenças sem
    importância prática: a leitura usa o PSI e a distância do K-S (de 0 a 1).
    """
    linhas = {}
    for coluna in colunas:
        ks = ks_2samp(referencia[coluna], atual[coluna])
        valor_psi = psi(referencia[coluna], atual[coluna])
        linhas[coluna] = {'PSI': valor_psi, 'K-S (distância)': ks.statistic, 'K-S (p-valor)': ks.pvalue,
                          'Leitura': leitura_psi(valor_psi)}
    return pd.DataFrame.from_dict(linhas, orient='index')


def erro_semana(real, previsto):
    """Erro de uma semana, depois que as vendas reais chegam: erro do dia (MAE, MAPE, MPE) e do total da rede."""
    real, previsto = np.asarray(real, dtype=float), np.asarray(previsto, dtype=float)
    return {'Dias de loja': len(real), 'MAE (€)': float(np.mean(np.abs(previsto - real))),
            'MAPE': float(np.mean(np.abs(previsto - real) / real)), 'MPE': float(np.mean((previsto - real) / real)),
            'Erro do total da rede': float(previsto.sum() / real.sum() - 1)}


def painel_semanas(real, previsto, semanas, sla_mape, sla_total_rede):
    """Erro de cada semana e a situação em relação ao SLA (MAPE do dia e erro do total da rede, em módulo)."""
    real, previsto = np.asarray(real, dtype=float), np.asarray(previsto, dtype=float)
    semanas = pd.Categorical(semanas)
    linhas = {}
    for semana in semanas.categories:
        na_semana = np.asarray(semanas == semana)
        if na_semana.any():
            linhas[semana] = erro_semana(real[na_semana], previsto[na_semana])
    painel = pd.DataFrame.from_dict(linhas, orient='index')
    fora = (painel['MAPE'] > sla_mape) | (painel['Erro do total da rede'].abs() > sla_total_rede)
    painel['Situação'] = np.where(fora, '🚨 fora do SLA', '✅ dentro do SLA')
    return painel
