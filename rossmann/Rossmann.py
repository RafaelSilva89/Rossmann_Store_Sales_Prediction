import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime


# Criando as classes Rossmann.py

class Rossmann(object):
    def __init__(self):
        self.home_path = '/Users/Rafael/Repos/6 - GitHub/1 - DS_em_Produção/'
        self.competition_distance_scaler = pickle.load(
            open(self.home_path + 'Parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_open_since_month = pickle.load(
            open(self.home_path + 'Parameter/competition_open_since_month.pkl', 'rb'))
        self.competition_open_since_year = pickle.load(
            open(self.home_path + 'Parameter/competition_open_since_year.pkl', 'rb'))
        self.year_scaler = pickle.load(open(self.home_path + 'Parameter/year_scaler.pkl', 'rb'))
        self.store_type_scaler = pickle.load(open(self.home_path + 'Parameter/store_type_scaler.pkl', 'rb'))

    def data_cleaning(self, df1):
        # PASSO 1 - Análise Descritiva dos Dados

        # 1.2. Colunas

        # Alterando as nomes das colunas
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear',
                    'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        snakecase = lambda x: inflection.underscore(x)

        cols_new = list(map(snakecase, cols_old))

        df1.columns = cols_new

        # 1.6. Filtrando os valores nulos

        # competition_distance
        df1.competition_distance.fillna(df1.competition_distance.mean(), inplace=True)

        # Preencher com a 0 os valores nulos
        cols_null = ['competition_open_since_month', 'competition_open_since_year',
                     'promo2_since_week', 'promo2_since_year', 'promo_interval']
        for i in cols_null:
            df1[i].fillna(0, inplace=True)

        # 1.7. Change Data Types

        # Mudando os tipos das colunas
        df1['date'] = pd.to_datetime(df1['date'])
        df1['competition_open_since_month'] = np.dtype('int64').type(df1['competition_open_since_month'])
        df1['competition_open_since_year'] = np.dtype('int64').type(df1['competition_open_since_year'])
        df1['promo2_since_week'] = np.dtype('int64').type(df1['promo2_since_week'])
        df1['promo2_since_year'] = np.dtype('int64').type(df1['promo2_since_year'])

        return df1

    def feature_engineering(self, df2):
        # PASSO 2 - Feature Engineering - (pré-processamento de dados)

        # Criando um atributo para ano
        df2['year'] = df2['date'].dt.year
        # Criando um atributo para mês
        df2['month'] = df2['date'].dt.month
        # Criando um atributo para dia
        df2['day'] = df2['date'].dt.day
        # Criando um atributo para semana do ano
        df2['week_of_year'] = df2['date'].dt.weekofyear
        # Criando um atributo ano e semana
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')
        # Mudando a classificação do assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else
        'extra' if x == 'b' else 'extended')
        # Mudando a classificação do state holiday
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else
        'easter_holiday' if x == 'b' else
        'christmas' if x == 'c' else 'regular_day')
        return df2

    def filtragem_das_variaveis(self, df3):
        # PASSO 3 - Filtragem das variáveis

        # 3.3. Selecão das Colunas
        cols_drop = ['open']
        df3 = df3.drop(cols_drop, axis=1)

        return df3

    def data_preparation(self, df5):
        # PASSO 05 - Modelagem dos dados

        # 5.3. Rescaling

        # competition distance (subscrevendo os valores em uma nova escala)
        df5['competition_distance'] = self.competition_distance_scaler.transform(df5[['competition_distance']].values)

        # competition open since_month (subscrevendo os valores em uma nova escala)
        df5['competition_open_since_month'] = self.competition_open_since_month.transform(
            df5[['competition_open_since_month']].values)

        # competition open since year (subscrevendo os valores em uma nova escala)
        df5['competition_open_since_year'] = self.competition_open_since_year.transform(
            df5[['competition_open_since_year']].values)

        # year
        df5['year'] = self.year_scaler.transform(df5[['year']].values)

        # 5.4. Transformação

        # state_holiday - One Hot Encoding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type - Label Encoding - não sabemos se existe uma ordem de importância
        df5['store_type'] = self.store_type_scaler.transform(df5['store_type'])

        # assortment - Ordinal Encoding - temos uma ordem de importância
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        # 5.6. Transformação de natureza cíclica

        # day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi / 7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi / 7)))

        # month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2. * np.pi / 12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2. * np.pi / 12)))

        # day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2. * np.pi / 30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2. * np.pi / 30)))

        # week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi / 52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi / 52)))

        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance',
                         'competition_open_since_month', 'competition_open_since_year', 'promo2',
                         'promo2_since_week', 'promo2_since_year', 'day_of_week_sin', 'day_of_week_cos',
                         'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos']

        return df5[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)

        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')