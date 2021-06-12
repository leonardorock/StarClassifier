import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("input/stars.csv", na_values=' ')

print(dataset)

X = dataset[["L","R"]].copy()
T = dataset[["Type"]].copy()

X_train, X_test, y_train, y_test = train_test_split(X, T, test_size=0.2, random_state=42)

clf = svm.NuSVR()
clf.fit(X_train, y_train)

print('Testes') 

Y = clf.predict(X_test)

# resultado 
print('Resultado procurado') 
print(y_test)
print("Score de treino: %f" % clf.score(X_train, y_train))
print('Resultado encontrado') 
print(Y)
print("Score do teste: %f" % clf.score(X_test, y_test))