import pandas as pd
import numpy as np
from sklearn import preprocessing
from grafica import dibuPtosRecta
#from matplotlib import pyplot as plt
from matplotlib import pylab as plt
from ClassPerceptron import Perceptron

# Leer distribucion.csv
datos = pd.read_csv("distribucion.csv")
nColum = list(datos.columns.values)

mini = datos['unidades'].min()
maxi = datos['unidades'].max()
datos['unidades']= (datos['unidades']-mini)/(maxi-mini)
mini = datos['descuento'].min()
maxi = datos['descuento'].max()
datos['descuento']= (datos['descuento']-mini)/(maxi-mini)




X = np.array(datos.iloc[:,0:2])
print(X)

#--- SALIDA BINARIA ---
T = datos['envio'] == 'normal'  #es boolean
T = np.array(T * 1)  #lo convierte en binario
print(T)

nCantEjemplos = X.shape[0] #n filas
nAtrib = X.shape[1] #n columnas


W = np.array([140,20])
b = 47

# W = np.array([1,0.13])
# b = 0.3125


# W = np.array([10,1])
# b = 1

# W = np.array([1,0.1])
# b = 0.45


ph=dibuPtosRecta(X,T, W, b, titulos = nColum[0:2])
plt.plot(X,X*W+b)

# # -- PESOS INICIALES - Determinan la ubicación de la recta
# W = np.array(np.random.uniform(-0.5,0.5,size=2))
# b = np.random.uniform(-0.5,0.5)
# ph=dibuPtosRecta(X,T, W, b, titulos = nColum[0:2])

# #--- parámetros del PERCEPTRON ---
# MAX_ITE = 1000
# alfa = 0.1
# ite=0

# # --- Entrenamiento del PERCEPTRON ---
# hubo_cambio=True
# while hubo_cambio and ite<MAX_ITE:
#     hubo_cambio=False
#     for e in range(X.shape[0]):
#         neta  = b + W[0]*X[e,0] + W[1]*X[e,1]
#         y = 1*(neta>0)
#         if y!=T[e]:
#             hubo_cambio=True
#             W[0] = W[0]+alfa*(T[e]-y)*X[e,0]
#             W[1] = W[1]+alfa*(T[e]-y)*X[e,1]
#             b += alfa*(T[e]-y)
#     ph = dibuPtosRecta(X,T, W, b,ph=ph)
#     ite += 1
