import pandas as pd
#import numpy as np
import matplotlib.pyplot as pl
from pandas import DataFrame as df
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder


#pegando os dados do csv, trocando brancos por NaN
df = pd.read_csv("input/stars.csv", na_values=' ')

# olhando alguns detalhes 
print(df.head())
print(df.describe())

#fazendo uma cópia
df2 = df.copy()

#removendo os NaN
df2.dropna(inplace=True)
print(df2.isna().sum())

#pegando a média sem os NaN
media_sem_nan = df2['Temperature'].mean()
mediana_sem_nan = df2['A_M'].median()

print(media_sem_nan, mediana_sem_nan)
#df['Temperature'] = df['Temperature'].replace(np.nan, media_sem_nan)
df['Temperature'] = df['Temperature'].fillna(media_sem_nan)
df['A_M'] = df['A_M'].fillna(mediana_sem_nan)

#pegando a média com os NaN trocados pela media_sem_nan
media = df['Temperature'].mean()
mediana = df['A_M'].median()
print(media, mediana)

#moda da cor de cabelo e cor de olho
print(df['Color'].mode())
print(df['Spectral_Class'].mode())

star_color = LabelEncoder()
star_color_types = star_color.fit_transform(df['Color'])
star_color_map = { index: label for index, label in enumerate(star_color.classes_) }

print(star_color_map)