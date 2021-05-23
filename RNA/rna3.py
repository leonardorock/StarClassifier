import numpy as np
import warnings as wr
from numpy import dtype, genfromtxt
from sklearn import neural_network as nn
from sklearn.exceptions import ConvergenceWarning

# taxa de aprendizado
lr = 0.5

X = genfromtxt(open('input/dataset.csv', 'r', encoding='utf-8-sig'), delimiter=';', dtype=float, missing_values='', filling_values=0)
T = genfromtxt(open('input/dataset_results.csv', 'r', encoding='utf-8-sig'), delimiter=';', dtype=float, missing_values='', filling_values=0)
Z = genfromtxt(open('test/test_data.csv', 'r', encoding='utf-8-sig'), delimiter=';', dtype=float, missing_values='', filling_values=0)
W = genfromtxt(open('test/test_data_results.csv', 'r', encoding='utf-8-sig'), delimiter=';', dtype=float, missing_values='', filling_values=0)

# dataset de treino
# lim letras   x--------------x  x--------------x  x--------------x  x--------------x  x--------------x  x--------------x
# X = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  #b
#               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  #C
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  #d
#               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  #e
#               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #F
#               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  #g
#               [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],  #h
#               [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  #I
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  #J
#               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]) #O

# rotulos do dataset de treino
# T = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

# dataset de testes
# lim letras   x--------------x  x--------------x  x--------------x  x--------------x  x--------------x  x--------------x
# Z = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  #b
#               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  #C
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  #d
#               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  #e
#               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #F
#               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  #g
#               [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],  #h
#               [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  #I
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  #J
#               [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]) #O
         
mlp = nn.MLPClassifier(hidden_layer_sizes=(40,), max_iter=240, alpha=1e-4, solver='sgd', verbose=10, random_state=1, learning_rate_init=lr)

# treino
print('#################### EXECUCAO ####################')
print('Treinamento') 
with wr.catch_warnings():
    wr.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X, T)

# teste
print('Testes') 
Y = mlp.predict(Z)

# resultado 
print('Resultado procurado') 
print(T)
print("Score de treino: %f" % mlp.score(X, T))
print('Resultado encontrado') 
print(Y)
print("Score do teste: %f" % mlp.score(Z, W))

sumY = [sum(Y[i]) for i in range(np.shape(Y)[0])] # saida
sumT = [sum(W[i]) for i in range(np.shape(W)[0])] # target

print('Comparacao de resultados') 
print(np.logical_xor(sumY, sumT))
