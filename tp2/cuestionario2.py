import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
from ClassPerceptron import Perceptron
from matplotlib import pylab as plt
import random

# Leer el archivo
datos = pd.read_csv("Mushroom.csv",encoding='iso8859-1')
notirar=['Tipo',' odor', ' spore-print-color', ' stalk-surface-below-ring',' gill-size',' bruises']
datos.drop(datos.columns.difference(notirar), axis=1, inplace=True)
mapeo = {'Tipo':{'p':1,'e':2},
         ' odor':{'a':1,'l':2,'c':3,'y':4,'f':5,'m':6,'n':7,'p':8,'s':9},
         ' spore-print-color':{'k':1,'n':2,'b':3,'h':4,'r':5,'o':6,'u':7,'w':8,'y':9},
         ' stalk-surface-below-ring':{'f':1,'y':2,'k':3,'s':4},
         ' gill-size':{'b':1,'n':2},
         ' bruises':{'t':1,'f':2}}
datos.replace(mapeo,inplace=True)


#--- DATOS DE ENTRENAMIENTO ---
X = np.array(datos.iloc[:,1:])
T = np.array((datos['Tipo'] == 1) * 1)
nColum = ['Otra', 'Blandos']


X_train, X_test, T_train, T_test = model_selection.train_test_split(
        X, T, test_size=0.30, random_state=42)


ppn = Perceptron(alpha=0.05, n_iter=100, draw=1, title=nColum, random_state=None,show_ite='LAST')
ppn.fit(X_train, T_train)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Iteraciones')
plt.ylabel('Cantidad de actualizaciones')
plt.show()

Y_test = ppn.predict(X_test)

aciertos = sum(Y_test == T_test)
print("aciertos = ", aciertos)

nAciertos = sum(Y_test == T_test)
print("%% de aciertos = %.2f %%" % (100*nAciertos/X_test.shape[0]))