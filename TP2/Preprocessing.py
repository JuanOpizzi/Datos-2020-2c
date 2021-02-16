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

df_data = pd.read_csv('https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq')
df_decision = pd.read_csv('https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0')
# -

columnsTitles=["tipo_de_sala","id_usuario","genero","edad","amigos",
               "parientes","precio_ticket",'nombre_sede','cant_acompañantes','volveria']

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


# Este preprossing esta basado en lo visto y hecho en el TP1, y es nuestro primero approach de preprossing

def preprod_tp1(df_datos, df_predict):
    df = pd.merge(df_datos, df_predict, how='inner', left_on='id_usuario', right_on='id_usuario')
    df = prepod_tp1_un_df(df)
    return df


def prepod_tp1_un_df(df):
    df = df.drop(['id_ticket', 'fila', 'nombre'], axis=1)
    df['cant_acompañantes'] = df['parientes'] + df['amigos']
    df = df.reindex(columns=columnsTitles)
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


df_decision.head()

df = preprod_tp1(df_data, df_decision)
df = replace_nulls_edad(df, 'media')
df.head()

df = normalizar_atributos_numericos(df)
df.head()

columnas_numericas = df.select_dtypes(include=numerics).columns.to_list()
df[columnas_numericas].max()

df.tipo_de_sala.value_counts()

df.tipo_de_sala.value_counts()

df = encodear_atributos_categoricos(df)
df.head()

df.shape


