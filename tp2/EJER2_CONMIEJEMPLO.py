import numpy as np
from grafica import *
import pandas as pd

datos = pd.read_csv("distribucion.csv")
nColum = list(datos.columns.values)

print(datos['unidades'])
print(datos['descuento'])

#NORMALIZAR POR MEDIA Y DESVIO (tipificacion)
media = datos['unidades'].mean()
desvio = datos['unidades'].std()
datos['unidades']= (datos['unidades']-media)/desvio

media = datos['descuento'].mean()
desvio = datos['descuento'].std()
datos['descuento']= (datos['descuento']-media)/desvio

print(datos['unidades'])
print(datos['descuento'])


entradas = np.array(datos.iloc[:,0:2])
salida = datos['envio'] == 'normal'  #es boolean
salida = np.array(salida * 1)  #lo convierte en binario

nCantEjemplos = entradas.shape[0] #n filas
nAtrib = entradas.shape[1] #n columnas


# W = np.array([140,20])
# b = 47

W = np.array([1,0.13])
b = 0.3125

# W = np.array([10,1])
# b = 1

# W = np.array([1,0.1])
# b = 0.45

# -- PESOS INICIALES - Determinan la ubicación de la recta
#W = np.array(np.random.uniform(-0.5,0.5,size=2))
#b = np.random.uniform(-0.5,0.5)


ph=dibuPtosRecta(entradas,salida, W, b, titulos = nColum[0:2])

# #--- parámetros del PERCEPTRON ---
# MAX_ITE = 1000
# alfa = 0.1
# ite=0

# # --- Entrenamiento del PERCEPTRON ---
# hubo_cambio=True
# while hubo_cambio and ite<MAX_ITE:
#     hubo_cambio=False
#     for e in range(nCantEjemplos):
#         neta  = b + W[0]*entradas[e,0] + W[1]*entradas[e,1]
#         y = 1*(neta>0)
#         if y!=salida[e]:
#             hubo_cambio=True
#             W[0] = W[0]+alfa*(salida[e]-y)*entradas[e,0]
#             W[1] = W[1]+alfa*(salida[e]-y)*entradas[e,1]
#             b += alfa*(salida[e]-y)
#     ph = dibuPtosRecta(entradas,salida, W, b,ph=ph)
#     ite += 1
#     print("ite %d" %ite)