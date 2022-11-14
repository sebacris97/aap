import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pylab as plt
from ClassPerceptron import Perceptron



#--- DATOS DE ENTRENAMIENTO ---
datosTrain = pd.read_csv("FrutasTrain.csv")
xTrain = np.array(datosTrain.iloc[:,0:2])
T_Train = np.array((datosTrain['Clase'] == 'Melon') * 1) 

#--- DATOS DE TESTEO ---
datosTest = pd.read_csv("FrutasTest.csv")
xTest = np.array(datosTest.iloc[:,0:2])
T_Test = np.array((datosTest['Clase'] == 'Melon') * 1) 

nColum = list(datosTrain.columns.values)



NORMALIZAR = "MAXMIN" #"STD" #"MAXMIN" #"NO"
if NORMALIZAR != "NO":
    if NORMALIZAR == "STD":
        normalizador = preprocessing.StandardScaler()
    elif NORMALIZAR == "MAXMIN":
        normalizador = preprocessing.MinMaxScaler()
    xTrain = normalizador.fit_transform(xTrain)
    xTest = normalizador.transform(xTest)



ORDEN = "ASC"
if ORDEN == "ASC":
    orden = np.argsort(T_Train)
if ORDEN == "DESC":
    orden = np.argsort(-1*T_Train)
else:
    orden = np.random.permutation(len(T_Train))
xTrain = xTrain[orden,:]
T_Train = T_Train[orden]


ppn = Perceptron(alpha=0.2, n_iter=100, draw=1, title=nColum[0:2], random_state=None, show_ite="LAST")
ppn.fit(xTrain, T_Train)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Iteraciones')
plt.ylabel('Cantidad de actualizaciones')
plt.show()

from PlotRegiones import plot_decision_regions

plot_decision_regions(xTrain, T_Train, classifier=ppn)
plt.xlabel(ppn.title[0])
plt.ylabel(ppn.title[1])
plt.legend(loc='lower left')
plt.show()

Y_Test = ppn.predict(xTest)
print("Y = ", Y_Test)
print("T = ", T_Test)
aciertos = sum(Y_Test == T_Test)
print("aciertos = ", aciertos)
nAciertos = sum(Y_Test == T_Test)
print("%% de aciertos = %.2f %%" % (100*nAciertos/xTest.shape[0]))



