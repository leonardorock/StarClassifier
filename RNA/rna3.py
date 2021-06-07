import numpy as np
import pandas as pd
import warnings as wr
from numpy import dtype, genfromtxt
from scipy.sparse import data
from sklearn import neural_network as nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

# taxa de aprendizado
lr = 0.003

dataset = pd.read_csv("input/stars.csv", na_values=' ')

print(dataset)

X = dataset[["L","R"]].copy()
T = dataset[["Type"]].copy()

X_train, X_test, y_train, y_test = train_test_split(X, T, test_size=0.35, random_state=42)
         
mlp = nn.MLPClassifier(hidden_layer_sizes=(256,128), max_iter=5000, alpha=1e-4, solver='adam', verbose=10, random_state=1, learning_rate_init=lr, activation='relu', n_iter_no_change=500)

# treino
print('#################### EXECUCAO ####################')
print('Treinamento') 
with wr.catch_warnings():
    wr.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

# teste
print('Testes') 
Y = mlp.predict(X_test)

# resultado 
print('Resultado procurado') 
print(y_test)
print("Score de treino: %f" % mlp.score(X_train, y_train))
print('Resultado encontrado') 
print(Y)
print("Score do teste: %f" % mlp.score(X_test, y_test))

# sumY = [sum(Y[i]) for i in range(np.shape(Y)[0])] # saida
# sumT = [sum(y_test[i]) for i in range(np.shape(y_test)[0])] # target

# print('Comparacao de resultados') 
# print(np.logical_xor(sumY, sumT))
