import numpy as np
import pandas as pd

column_titles = ["tipo_de_sala","id_usuario","genero","edad","amigos",
                 "parientes","precio_ticket",'nombre_sede','cant_acompañantes','volveria']

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


def eliminar_features_que_no_aportan_info(df, drop_fila = False):
    if drop_fila == True:
        df = df.drop(['id_ticket', 'fila', 'nombre'], axis=1)
    else:
        df = df.drop(['id_ticket', 'nombre'], axis=1)
    return df

def crear_feature_acompañantes(df):
    df['cant_acompañantes'] = df['parientes'] + df['amigos']
    return df

# reemplaza los valores nulos de la columna que se pasa
# si la columna no existe en el data set no hace nada
def replace_nulls_column(df, columna, metrica):
    if columna not in df.columns:
        return
    else:
        s = df[columna]
        if metrica == 'moda':
            df = df.replace({columna: np.nan}, s.mode())
        elif metrica == 'mediana':
            df = df.replace({columna: np.nan}, s.median())
        elif metrica == 'media':
            df = df.replace({columna: np.nan}, s.mean())
        df = df.round({columna: 1})
        return df

# con dummy_na pasamos nan a categoria y con drop_first eliminamos una columna, que a fines
# practicos es informacion redundante
def encodear_atributos_categoricos(df):
    encodeado = pd.get_dummies(df, columns=['tipo_de_sala', 'genero', 'nombre_sede'], dummy_na=True, drop_first=True)
    return encodeado

def normalizar_atributos_numericos(df):
    columnas_numericas = df.select_dtypes(include=numerics).columns.to_list()
    maximo = df[columnas_numericas].max()
    minimo = df[columnas_numericas].min()
    df[columnas_numericas] = (df[columnas_numericas] - df[columnas_numericas].min()) / (df[columnas_numericas].max() - df[columnas_numericas].min())
    return df