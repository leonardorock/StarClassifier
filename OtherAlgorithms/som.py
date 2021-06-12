import pandas as pd
from sklearn_som.som import SOM

dataset = pd.read_csv("input/stars.csv", na_values=' ')

print(dataset)

X = dataset[["L","R"]].copy()

som = SOM(m=3, n=1, dim=2)
som.fit(X)

Y = som.predict(X)

print(Y)

