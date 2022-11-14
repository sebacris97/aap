import pandas as pd
import numpy as np
from matplotlib import pylab as plt

from sklearn import model_selection,preprocessing

from ClassPerceptron import Perceptron

# Leer el archivo
datos = pd.read_csv("Iris.csv")
nColum = list(datos.columns.values)
print(nColum)

#--- DATOS DE ENTRENAMIENTO ---
X = np.array(datos.iloc[:,:-1])
T = np.array((datos['class'] == 'Iris-setosa') * 1)
nColum = ['Otra', 'Iris-setosa']

#--- CONJUNTOS DE ENTRENAMIENTO Y TESTEO ---
X_train, X_test, T_train, T_test = model_selection.train_test_split(
        X, T, test_size=0.4, random_state=42)

normalizarEntrada = 1  # 1 si normaliza; 0 si no
if normalizarEntrada:
    #--- Normalización lineal entre 0 y 1 ---
    # normalizador = preprocessing.MinMaxScaler()

    # Normaliza utilizando la media y el desvio
    normalizador= preprocessing.StandardScaler()
    
    X_train = normalizador.fit_transform(X_train)
    X_test  = normalizador.transform(X_test)

ppn = Perceptron(alpha=0.01, n_iter=650, random_state=None).fit(X_train, T_train)


Y_test = ppn.predict(X_test)
aciertos = sum(Y_test == T_test)
print("aciertos = ", aciertos)
nAciertos = sum(Y_test == T_test)
print("%% de aciertos = %.2f %%" % (100*nAciertos/X_test.shape[0]))