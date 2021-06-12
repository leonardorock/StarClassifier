import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("input/stars.csv", na_values=' ')

print(dataset)

X = dataset[["L","R", "A_M"]].copy()
T = dataset[["Type"]].copy()

X_train, X_test, y_train, y_test = train_test_split(X, T, test_size=0.2, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=6, algorithm='kd_tree', leaf_size=30, n_jobs=4, weights='distance')
neigh.fit(X_train, y_train)

print('Testes') 

Y = neigh.predict(X_test)

# resultado 
print('Resultado procurado') 
print(y_test)
print("Score de treino: %f" % neigh.score(X_train, y_train))
print('Resultado encontrado') 
print(Y)
print("Score do teste: %f" % neigh.score(X_test, y_test))