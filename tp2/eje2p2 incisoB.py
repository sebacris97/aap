import pandas as pd
import numpy as np
from sklearn import preprocessing
from ClassPerceptron import Perceptron
from matplotlib import pyplot as plt

datos = pd.read_csv("hojas.csv")
dic = {'Perimetro':770,'Area':5000,'Clase':'Hoja'}
datos = pd.concat([datos, pd.DataFrame([dic])])

"""
#DIVIDR DATAFRAME EN 2
split=np.array_split(datos, 2)
datosTrain=split[0]
datosTest=split[1]
"""

#--- DATOS DE ENTRENAMIENTO ---
xTrain = np.array(datos.iloc[:,0:2])
T_Train = np.array((datos['Clase'] == 'Helecho') * 1) 
#--- DATOS DE TESTEO --- 
#en este caso son pocos datos asi que usamos los mismos
xTest = xTrain
T_Test = T_Train


nColum = list(datos.columns.values)
NORMALIZACIONES = ["SIN NORMALIZAR","MAXMIN","STD"]
ORDENES = ["SIN ORDEN","ASC","DESC"]
ALPHAS = [0.2,0.005]


NORMALIZAR = "STD"
if NORMALIZAR != "SIN NORMALIZAR":
    if NORMALIZAR == "MAXMIN":
        normalizador = preprocessing.MinMaxScaler()
    elif NORMALIZAR == "STD":
        normalizador = preprocessing.StandardScaler()
    xTrain = normalizador.fit_transform(xTrain)
    xTest = normalizador.transform(xTest)

        
ORDEN = "SIN ORDEN"
if ORDEN == "ASC":
    orden = np.argsort(T_Train)
if ORDEN == "DESC":
    orden = np.argsort(-1*T_Train)
else:
    orden = np.random.permutation(len(T_Train))
xTrain = xTrain[orden,:]
T_Train = T_Train[orden]

ppn = Perceptron(alpha=0.01, n_iter=300, draw=1, title=nColum[0:2], random_state=None)
OUT = ppn.fit(xTrain, T_Train)
print("w: ",ppn.w_) #promedio de 5 ejecuciones: 429,432 y -76,018
print("b: ",ppn.b_)                            #0,334
