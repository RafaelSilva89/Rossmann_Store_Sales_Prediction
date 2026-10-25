"""Funções de gráficos e de formatação usadas no notebook. Cada gráfico é salvo na pasta Imagens/ do projeto."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PASTA_IMAGENS = Path(__file__).resolve().parents[1] / 'Imagens'

COR_PRINCIPAL = '#1F4E79'
COR_DESTAQUE = '#C0392B'
COR_NEUTRA = '#B0B7C3'


def salvar(nome_arquivo):
    plt.savefig(PASTA_IMAGENS / f'{nome_arquivo}.png', bbox_inches='tight', dpi=110)


def titulo(texto, subtitulo=None):
    """Título e subtítulo alinhados à esquerda, acima da área do gráfico (espaçamento em polegadas)."""
    altura = plt.gcf().get_figheight()
    topo = plt.gcf().subplotpars.top
    plt.figtext(0.01, topo + 0.45 / altura, texto, ha='left', va='bottom', fontsize=16)
    if subtitulo:
        plt.figtext(0.01, topo + 0.15 / altura, subtitulo, ha='left', va='bottom', fontsize=11, color='#656565',
                    style='italic')


def pct(valor, casas=0):
    """0.1234 -> '12%' (ou '12,3%' com casas=1)."""
    return f'{valor:.{casas}%}'.replace('.', ',')


def euros(valor, casas=0):
    """7088.4 -> '€ 7.088' (separador de milhar com ponto). As vendas da Rossmann (Alemanha) estão em euros."""
    texto = f'{valor:,.{casas}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
    return f'€ {texto}'


def plot_barras_h(serie, titulo_grafico, nome_arquivo, subtitulo=None, formato=pct, destaque=None, referencia=None):
    """Barras horizontais ordenadas, com o valor escrito em cada barra.

    destaque: rótulos pintados com a cor de destaque. referencia: valor desenhado como linha tracejada.
    """
    serie = serie.sort_values()
    cores = [COR_DESTAQUE if destaque and rotulo in destaque else COR_PRINCIPAL for rotulo in serie.index]
    plt.figure(figsize=(10, max(4, 0.4 * len(serie))))
    barras = plt.barh(serie.index.astype(str), serie.values, color=cores)
    plt.bar_label(barras, labels=[formato(v) for v in serie.values], padding=3, fontsize=9)
    if referencia is not None:
        plt.axvline(referencia, color='black', ls='--', lw=1)
        plt.annotate(f'geral: {formato(referencia)}', (referencia, len(serie) - 0.4), xytext=(4, 0),
                     textcoords='offset points', fontsize=9)
    plt.xticks([])
    sns.despine(left=True, bottom=True)
    titulo(titulo_grafico, subtitulo)
    salvar(nome_arquivo)
    plt.show()


def plot_boxplot_grupos(df, coluna_grupo, coluna_valor, titulo_grafico, nome_arquivo, subtitulo=None, ordem=None,
                        limite_y=None, formato=euros):
    """Boxplot do valor por grupo, com a mediana escrita acima de cada caixa (sem os pontos extremos)."""
    ordem = ordem or df.groupby(coluna_grupo)[coluna_valor].median().sort_values(ascending=False).index.tolist()
    plt.figure(figsize=(max(7, 1.6 * len(ordem)), 5))
    ax = sns.boxplot(data=df, x=coluna_grupo, y=coluna_valor, order=ordem, color=COR_PRINCIPAL, showfliers=False,
                     boxprops={'alpha': 0.75})
    medianas = df.groupby(coluna_grupo)[coluna_valor].median()
    for i, grupo in enumerate(ordem):
        ax.annotate(f'mediana {formato(medianas[grupo])}', (i, medianas[grupo]), xytext=(0, 6),
                    textcoords='offset points', ha='center', fontsize=9, color='#222222',
                    bbox={'boxstyle': 'round,pad=0.2', 'fc': 'white', 'ec': 'none', 'alpha': 0.85})
    if limite_y:
        ax.set_ylim(0, limite_y)
    ax.set(xlabel='', ylabel=coluna_valor)
    sns.despine()
    titulo(titulo_grafico, subtitulo)
    salvar(nome_arquivo)
    plt.show()
