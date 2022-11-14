# -*- coding: utf-8 -*-
"""MLP_XOR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_hM9dwQIPGvQfqgq0CAz8uC_1AOIilHH
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np
import time
#from matplotlib import pyplot as plt
from matplotlib import pylab as plt
from IPython import display

from graficaMLP import dibuPtos_y_2Rectas
from Funciones import evaluar, evaluarDerivada

X = np.array([ [-1, -1], [-1, 1], [1, -1], [1, 1]])
Y = np.array([-1, 1, 1, -1]).reshape(-1,1)

nFilas = X.shape[0]
entradas = X.shape[1]
ocultas = 6
salidas = Y.shape[1]

W1 = np.random.uniform(-1,1,[ocultas, entradas])
b1 = np.random.uniform(-1,1, [ocultas,1])
W2 = np.random.uniform(-1,1,[salidas, ocultas])
b2 = np.random.uniform(-1,1, [salidas,1])
        
#dibuPtos_y_2Rectas(X,Y, W1, b1)

FunH = 'sigmoid'
FunO = 'tanh'

alfa = 0.5
CotaError = 1.0e-06
MAX_ITERA = 500
ite = 0
errorAnt = 0
AVGError = 1
errores = []
ph=0
while ( abs(AVGError-errorAnt) > CotaError ) and ( ite < MAX_ITERA ):
    errorAnt = AVGError
    AVGError = 0
    for e in range(nFilas):  #para cada ejemplo

        xi = X[e:e+1, :]     # ejemplo a ingresar a la red

        # propagar el ejemplo hacia adelante
        netasH = W1 @ xi.T + b1
        salidasH = evaluar(FunH, netasH)
        netasO = W2 @ salidasH + b2
        salidasO = evaluar(FunO, netasO)

        # calcular los errores en ambas capas        
        ErrorSalida = Y[e]-salidasO
        deltaO = ErrorSalida * evaluarDerivada(FunO,salidasO)
        deltaH = evaluarDerivada(FunH,salidasH)*(W2.T @ deltaO)

        # corregir todos los pesos      
        W1 = W1 + alfa * deltaH @ xi 
        b1 = b1 + alfa * deltaH 
        W2 = W2 + alfa * deltaO @ salidasH.T 
        b2 = b2 + alfa * deltaO 

        AVGError = AVGError + np.mean(ErrorSalida**2)
    
    AVGError = AVGError / nFilas
    errores.append(AVGError)
    
    ite = ite + 1
    
    # Graficar las rectas
    if (ite % 10) ==0:
        ph = dibuPtos_y_2Rectas(X,Y, W1, b1, ph)

plt.plot(range(1, len(errores) + 1), errores, marker='o')
plt.xlabel('Iteraciones')
plt.ylabel('ECM')
plt.show()

