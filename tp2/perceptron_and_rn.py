# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pylab as plt
from ClassPerceptron import Perceptron
import pandas as pd

df = pd.read_csv('distribucion.csv')
print(df)

# Ejemplos de entrada de la función AND
X = np.array([[0,0], [0,1],[1,0],[1,1]])
T = np.array([0,0,0,1])

ppn = Perceptron(alpha=0.01, n_iter=30, draw=1, title=['X1', 'X2'], random_state=1)
# --- utilice random_state=None para que los pesos se inicializacen en forma aleatoria --
ppn.fit(X, T)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Iteraciones')
plt.ylabel('Cantidad de actualizaciones')
plt.show()

from PlotRegiones import plot_decision_regions

plot_decision_regions(X, T, classifier=ppn)
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper right')
plt.show()
plt.close()

Y = ppn.predict(X)
print("Y = ", Y)
print("T = ", T)
aciertos = sum(Y == T)
print("aciertos = ", aciertos)
nAciertos = sum(Y==T)
print("%% de aciertos = %.2f %%" % (100*nAciertos/X.shape[0]))
