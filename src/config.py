"""Constantes do projeto: caminhos, semente, colunas, datas de corte e premissas."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO_VENDAS = RAIZ / 'Datasets' / 'train.csv'
ARQUIVO_LOJAS = RAIZ / 'Datasets' / 'store.csv'
ARQUIVO_FUTURO = RAIZ / 'Datasets' / 'test.csv'   # as 6 semanas seguintes (ago-set/2015), sem as vendas
PASTA_MODELO = RAIZ / 'Model'
PASTA_REPORTS = RAIZ / 'reports'

SEMENTE = 42
ALVO = 'vendas'
N_JOBS = 2   # a máquina tem 8 GB de RAM: poucos processos em paralelo

# Tradução das colunas do Kaggle para português
COLUNAS = {
    'Store': 'loja', 'DayOfWeek': 'dia_semana', 'Date': 'data', 'Sales': 'vendas', 'Customers': 'clientes',
    'Open': 'aberta', 'Promo': 'promo', 'StateHoliday': 'feriado', 'SchoolHoliday': 'feriado_escolar',
    'StoreType': 'tipo_loja', 'Assortment': 'sortimento', 'CompetitionDistance': 'distancia_concorrente',
    'CompetitionOpenSinceMonth': 'concorrente_desde_mes', 'CompetitionOpenSinceYear': 'concorrente_desde_ano',
    'Promo2': 'promo2', 'Promo2SinceWeek': 'promo2_desde_semana', 'Promo2SinceYear': 'promo2_desde_ano',
    'PromoInterval': 'promo2_meses', 'Id': 'id',
}
FERIADOS = {'0': 'nenhum', 'a': 'público', 'b': 'Páscoa', 'c': 'Natal'}
SORTIMENTOS = {'a': 'básico', 'b': 'extra', 'c': 'estendido'}

# Distância usada quando não há concorrente informado (o maior valor da base é 75.860 m)
DISTANCIA_SEM_CONCORRENTE = 100_000

# Separação temporal: o teste são as mesmas 6 últimas semanas usadas pela versão original
SEMANAS_PREVISAO = 6
INICIO_VALIDACAO = '2015-05-08'
INICIO_TESTE = '2015-06-19'

CATEGORICAS = ['feriado', 'tipo_loja', 'sortimento']
NUMERICAS = ['dia_semana', 'promo', 'feriado_escolar', 'distancia_concorrente', 'sem_concorrente',
             'meses_concorrente', 'concorrente_data_desconhecida', 'promo2', 'semanas_promo2', 'promo2_mes_ativo',
             'ano', 'mes', 'dia', 'semana_ano']
# Médias da loja: calculadas só com o período de treino (transformador MediasDaLoja, em src/modelos.py)
MEDIAS_LOJA = ['media_loja', 'media_loja_dia_semana', 'media_loja_promo']
# Entram no modelo: a loja é usada só para calcular as médias (o número da loja em si não tem ordem)
FEATURES = ['loja'] + CATEGORICAS + NUMERICAS
# Vazamento: o número de clientes do dia só é conhecido depois que o dia acaba
PROIBIDAS = ['clientes']
