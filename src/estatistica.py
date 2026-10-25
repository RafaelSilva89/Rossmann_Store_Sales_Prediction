"""Testes estatísticos das hipóteses da EDA, sempre com uma medida de tamanho de efeito.

Com milhares de linhas, quase tudo vira "significativo" (p pequeno). Por isso, o que importa na
leitura é o TAMANHO do efeito (V de Cramér, odds ratio, correlação ponto-bisserial), não só o p.
"""
import numpy as np
import pandas as pd
from scipy import stats


def cramers_v(x, y):
    """V de Cramér (0 = nenhuma associação, 1 = associação total), com correção de viés de Bergsma."""
    tabela = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(tabela)[0]
    n = tabela.to_numpy().sum()
    r, k = tabela.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c = r - (r - 1) ** 2 / (n - 1)
    k_c = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(min(k_c - 1, r_c - 1), 1e-12)))


def teste_qui_quadrado(x, y):
    """Qui-quadrado de independência entre duas categóricas + V de Cramér."""
    chi2, p, gl, _ = stats.chi2_contingency(pd.crosstab(x, y))
    return {'teste': 'Qui-quadrado', 'estatística': chi2, 'gl': gl, 'p-valor': p, 'efeito': 'V de Cramér',
            'tamanho do efeito': cramers_v(x, y)}


def teste_mann_whitney(grupo_a, grupo_b):
    """Mann-Whitney (as distribuições de dois grupos diferem?) + correlação ponto-bisserial de postos.

    O tamanho do efeito vai de -1 a 1: 0 = sem diferença; 0,3 já é um efeito moderado.
    """
    u, p = stats.mannwhitneyu(grupo_a, grupo_b, alternative='two-sided')
    efeito = 2 * u / (len(grupo_a) * len(grupo_b)) - 1  # positivo quando o grupo_a tende a ter valores maiores
    return {'teste': 'Mann-Whitney', 'estatística': u, 'p-valor': p, 'efeito': 'correlação de postos',
            'tamanho do efeito': efeito}


def odds_ratio(condicao, alvo):
    """Odds ratio de uma tabela 2x2: quantas vezes a chance do alvo é maior quando a condição é verdadeira."""
    tabela = pd.crosstab(pd.Series(condicao, name='condicao').astype(bool), pd.Series(alvo, name='alvo').astype(bool))
    a, b = tabela.loc[True, True], tabela.loc[True, False]
    c, d = tabela.loc[False, True], tabela.loc[False, False]
    return float((a * d) / (b * c))


def regra_iqr(serie):
    """Limites da regra do IQR (Q1 - 1,5·IQR e Q3 + 1,5·IQR) e quantos valores ficam fora deles."""
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    fora = (serie < inferior) | (serie > superior)
    return {'limite inferior': inferior, 'limite superior': superior, 'outliers': int(fora.sum()),
            '% outliers': float(fora.mean())}


def teste_kruskal(valores, grupos):
    """Kruskal-Wallis (a distribuição difere entre 3 ou mais grupos?) + épsilon² como tamanho de efeito.

    Épsilon² vai de 0 a 1: 0,01 = pequeno, 0,08 = médio, 0,26 ou mais = grande.
    """
    dados = pd.DataFrame({'valor': valores, 'grupo': grupos}).dropna()
    amostras = [g['valor'].to_numpy() for _, g in dados.groupby('grupo', observed=True)]
    h, p = stats.kruskal(*amostras)
    n = len(dados)
    return {'teste': 'Kruskal-Wallis', 'estatística': h, 'p-valor': p, 'efeito': 'épsilon²',
            'tamanho do efeito': h * (n + 1) / (n ** 2 - 1)}


def teste_spearman(x, y):
    """Correlação de Spearman (relação monotônica entre duas numéricas): de -1 a 1, o próprio tamanho do efeito."""
    dados = pd.DataFrame({'x': x, 'y': y}).dropna()
    rho, p = stats.spearmanr(dados['x'], dados['y'])
    return {'teste': 'Spearman', 'estatística': rho, 'p-valor': p, 'efeito': 'rho', 'tamanho do efeito': rho}


def teste_wilcoxon(antes, depois):
    """Wilcoxon pareado (a mesma unidade mudou entre dois momentos?) + mediana da diferença como efeito."""
    dados = pd.DataFrame({'antes': antes, 'depois': depois}).dropna()
    w, p = stats.wilcoxon(dados['depois'], dados['antes'])
    return {'teste': 'Wilcoxon pareado', 'estatística': w, 'p-valor': p, 'efeito': 'mediana da diferença',
            'tamanho do efeito': float((dados['depois'] - dados['antes']).median())}
