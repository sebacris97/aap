import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
from ClassPerceptron import Perceptron
from matplotlib import pylab as plt
import random

# Leer el archivo
datos = pd.read_csv("zoo.csv",encoding='iso8859-1')
del datos["animal"]

df = datos.groupby(['Clase'], sort=False).size().reset_index(name='Count')


#--- DATOS DE ENTRENAMIENTO ---
X = np.array(datos.iloc[:,0:-1])
opciones=datos['Clase'].unique()
rand = random.choice(opciones)
print(rand)
salida=datos['Clase']==rand
salida=np.array(salida*1)
T = np.array((datos['Clase'] == 'Ave') * 1)
nColum = ['Otra', 'Mamifero']

X_train, X_test, T_train, T_test = model_selection.train_test_split(
        X, T, test_size=0.30, random_state=42)



ppn = Perceptron(alpha=0.05, n_iter=200, draw=True, title=nColum)
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

