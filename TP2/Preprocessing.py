# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Preprocessing TP1
# Aca se escriben los distintos preprocesados y se describe rapidamente su funcionalidad
#

# +
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
np.warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
import category_encoders as ce

sns.set()


# -

column_titles = ["tipo_de_sala","id_usuario","genero","edad","amigos",
                 "parientes","precio_ticket",'nombre_sede','cant_acompañantes','volveria']

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


# Este preprossing esta basado en lo visto y hecho en el TP1, y es nuestro primero approach de preprossing

def generate_initial_dataset(data_path, decision_path):
    df_datos = pd.read_csv(data_path)
    df_predict = pd.read_csv(decision_path)
    df = pd.merge(df_datos, df_predict, how='inner', left_on='id_usuario', right_on='id_usuario')
    df = df.drop(['id_ticket', 'fila', 'nombre'], axis=1)
    df['cant_acompañantes'] = df['parientes'] + df['amigos']
    df = df.reindex(columns = column_titles)
    return df


def generate_holdout_dataset(df_path):
    df = pd.read_csv(df_path)
    df = df.drop(['id_ticket', 'fila', 'nombre'], axis=1)
    df['cant_acompañantes'] = df['parientes'] + df['amigos']
    df = df.reindex(columns = column_titles)
    return df


# Reemplazo los valores nulos de edad con la moda, la media o la meediana, segun que quiera

def replace_nulls_edad(df, metrica):
    s = df['edad']
    if metrica == 'moda':
        df = df.replace({'edad': np.nan}, s.mode())
    elif metrica == 'mediana':
        df = df.replace({'edad': np.nan}, s.median())
    elif metrica == 'media':
        df = df.replace({'edad': np.nan}, s.mean())
    df = df.round({'edad': 1})
    return df


# Encodeo todos los atributos categoricos

def encodear_atributos_categoricos(df):
    encoder = ce.BinaryEncoder(cols=['tipo_de_sala', 'genero', 'nombre_sede'],return_df=True)
    df = encoder.fit_transform(df) 
    return df


# Agarro todos los atributos numericos y los normalizo

def normalizar_atributos_numericos(df):
    columnas_numericas = df.select_dtypes(include=numerics).columns.to_list()
    maximo = df[columnas_numericas].max()
    minimo = df[columnas_numericas].min()
    df[columnas_numericas] = (df[columnas_numericas] - df[columnas_numericas].min()) / (df[columnas_numericas].max() - df[columnas_numericas].min())
    return df






