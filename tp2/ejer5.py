import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
from ClassPerceptron import Perceptron
from matplotlib import pylab as plt
import grafica


#sepallength,sepalwidth,petallength,petalwidth,class

# Leer el archivo
datos = pd.read_csv("Iris.csv")

#--- DATOS DE ENTRENAMIENTO ---
X = np.array(datos.iloc[:,:-1])
salida=datos['class']=='Iris-setosa'
salida=np.array(salida*1)
T = np.array((datos['class'] == 'Iris-setosa') * 1)
nColum = ['Otra', 'Iris-setosa']


X_train, X_test, T_train, T_test = model_selection.train_test_split(
                                   X, T, test_size=0.30, random_state=None)
normalizador = preprocessing.StandardScaler()
X_train = normalizador.fit_transform(X_train)
X_test = normalizador.transform(X_test)

ppn = Perceptron(alpha=0.1, n_iter=50, draw=1, title=nColum, random_state=None)
ppn.fit(X_train, T_train)

plt.scatter(X,T.reshape(-1,1))
    

Y_test = ppn.predict(X_test)

aciertos = sum(Y_test == T_test)
print("aciertos = ", aciertos)

nAciertos = sum(Y_test == T_test)
print("%% de aciertos = %.2f %%" % (100*nAciertos/X_test.shape[0]))