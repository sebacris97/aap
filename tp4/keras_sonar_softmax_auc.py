import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing, metrics, model_selection

from sklearn.model_selection import cross_val_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

df = pd.read_csv('Iris.csv')
nomClases = pd.unique(df['class'])

# Tomamos todas las columnas menos la última
X = np.array(df.iloc[:, 0:-1])
Y = np.array(df.iloc[:,-1])

le = preprocessing.LabelEncoder()
Y = le.fit_transform(Y)

# creating instance of one-hot-encoder
#enc = preprocessing.OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)
#Y = enc.fit_transform(Y.reshape(-1,1)).toarray()

entradas = X.shape[1]
ocultas = 4

#-- la red tendrá una salida para cada tipo de flor
salidas = len(np.unique(Y))
print("entradas = %d ; salidas = %d" % (entradas, salidas))

#--- CONJUNTOS DE ENTRENAMIENTO Y TESTEO ---
X_train, X_test, Y_train, Y_test = model_selection.train_test_split( \
        X,Y, test_size=0.30)#, random_state=42)
Y_trainB = to_categorical(Y_train)
    
normalizarEntrada = 1  # 1 si normaliza; 0 si no

if normalizarEntrada:
    # Escala los valores entre 0 y 1
    min_max_scaler = preprocessing.StandardScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

model = Sequential()
model.add(Dense(ocultas, input_shape=[entradas], activation='tanh'))
model.add(Dense(salidas, activation='sigmoid'))

model.summary()  #-- muestra la cantidad de parámetros de la red

# Configuración para entrenamiento
#-- se utilizará SGD (descenso de gradiente esticástico),
#-- MSE (error cuadrático medio) y ACCURACY como medida de performance
model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics='accuracy')

model.fit(X_train,Y_trainB, epochs=500) #, batch_size=20)

# predecir la salida del modelo
Y_pred = model.predict(X_train)
Y_pred_nro = np.argmax(Y_pred,axis=1)  #-- conversión a entero

print("%% aciertos X_train : %.3f" % metrics.accuracy_score(Y_train, Y_pred_nro))

report = metrics.classification_report(Y_train, Y_pred_nro)
print("Confusion matrix Training:\n%s" % report) 

MM = metrics.confusion_matrix(Y_train, Y_pred_nro)
print("Confusion matrix:\n%s" % MM)

#--- AUC ---
clase_positiva=0  #-- la 1ra. columna corresponde a MINE
Y_pred = model.predict(X_train)
Y_proba = Y_pred[:,clase_positiva]

Y_positivo = Y_trainB[:,clase_positiva]
fpr, tpr, threshold = metrics.roc_curve(Y_positivo, Y_proba)
roc_auc = metrics.auc(fpr, tpr)

#Generamos un clasificador sin entrenar , que asignará 0 a todo
siempre_0 = np.zeros(Y_positivo.shape)
ns_fpr, ns_tpr, _ = metrics.roc_curve(Y_positivo,siempre_0)


# method I: plt
plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Sin entrenar')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

