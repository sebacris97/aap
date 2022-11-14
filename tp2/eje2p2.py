import pandas as pd
import numpy as np
from sklearn import preprocessing
from ClassPerceptron import Perceptron
from matplotlib import pyplot as plt

datos = pd.read_csv("hojas.csv")
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

    
for i in NORMALIZACIONES:
    NORMALIZAR = i
    if NORMALIZAR != "SIN NORMALIZAR":
        if NORMALIZAR == "MAXMIN":
            normalizador = preprocessing.MinMaxScaler()
        elif NORMALIZAR == "STD":
            normalizador = preprocessing.StandardScaler()
        xTrain = normalizador.fit_transform(xTrain)
        xTest = normalizador.transform(xTest)
        
    for j in ORDENES:
        
        ORDEN = j
        if ORDEN == "ASC":
            orden = np.argsort(T_Train)
        if ORDEN == "DESC":
            orden = np.argsort(-1*T_Train)
        else:
            orden = np.random.permutation(len(T_Train))
        xTrain = xTrain[orden,:]
        T_Train = T_Train[orden]
        
        for k in ALPHAS:
            
            alcanzo = 0
            iteraciones = []
            for exc in range(50):
                ppn = Perceptron(alpha=k, n_iter=100, draw=0, title=nColum[0:2], random_state=None)
                trials = ppn.fit(xTrain, T_Train)[1]
                plt.clf()
                if trials<100:
                    alcanzo += 1
                    iteraciones.append(trials)
            print("NORMALIZACION: ",i)
            print("ORDEN: ",j)
            print("ALPHA: ",k)
            print("PORCENTAJE DE FINALIZACION: ",alcanzo/100)
            promedio = sum(iteraciones)/len(iteraciones)
            print("PROMEDIO DE ITERACIONES QUE NECESITO CUANDO FINALIZO: ",promedio)
            print("\n\n")
