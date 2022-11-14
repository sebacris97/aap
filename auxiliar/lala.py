from sklearn import metrics
import tensorflow

Y_train = [0,1,2,3,0,1,2,3,0,1,2,3]
Y_pred  = [0,2,1,3,0,1,2,0,0,1,2,3]
MM = metrics.confusion_matrix(Y_train,Y_pred)
print("matriz de confusion:\n%s" %(MM))

y_train=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,
2,2,2,2,2,2,2,2,2,2,2,2,5,5,5,5,5,5,5,5,
3,3,3,3,3,3,3,3,3,3,3,3,1,
4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,1]



y_pred=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
3,3,3,3,3,3,3,3,3,3,3,3,3,
4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]

MM = metrics.confusion_matrix(y_train,y_pred)
print("matriz de confusion:\n%s" %(MM))
aciertos = metrics.accuracy_score(y_train,y_pred)
#porcentaje de a cuantos le pego, o cuantos coinciden de los calculados contra
#lo que se predecia, si le pego 
print(aciertos)

reporte = metrics.classification_report(y_train,y_pred)
print(reporte)

df = pd.read_csv('semillas.csv')
print(df.shape)
print(df['Clase'].value_counts())
print(df['Clase'].nunique()) #cuantas clases diferentes hay osea neuronas de salida

"""
a) Con respecto a la arquitectura, indique:

▪ La cantidad de neuronas de la capa de entrada. 8 de entardas + 1 bias = 9

▪ La cantidad de neuronas de la capa de salida. 3 de salida pues son 3 clases

▪ La cantidad de pesos (arcos) que tiene la red si se utiliza una única
capa oculta formada por 4 neuronas

a ver, son 8 neuronas de entrada + 1 bias, conectadas 4 neuronas
que a su vez estan conectadas 3 neuronas de salida + 1 bias,
tenemos entonces 9*4 + 5 * 3 = 51 arcos

"""


#############################

"""
El archivo Sonar.csv contiene registros de rebotes de señales de sonar tomadas
en varios ángulos y bajo distintas condiciones.
La tarea es utilizar una red multiperceptrón para discriminar entre señales
de sonar rebotadas en un cilindro de metal y aquellas rebotadas en una
roca más o menos cilíndrica.
Cada muestra es un conjunto de 60 números en el rango de 0 a 1.
Cada número representa la energía, dentro de una banda de
frecuencia particular, integrada durante un cierto período de tiempo.
La etiqueta asociada a cada registro contiene "Rock" si el objeto es una roca
y "Mine" si es una mina (cilindro de metal).

Si se utilizan 6 neuronas ocultas ¿cuántos pesos (arcos) tiene la red?


La red tendrá (60+1)*6 + (6+1)*2= 61*6+7*2 = 366+14=380 arcos.
Cada neurona de la capa oculta recibe 61 arcos
(60 corresponden a los datos del ejemplo y 1 al bias)
y cada neurona de la capa de salida recibe 7 arcos
(6 conectados a las neuronas de la capa oculta y 1 al bias de la capa de salida). Se trata de un problema de 2 clases por lo que tendrá 2 neuronas a la salida.

Las respuestas son: 380


"""

"""
178 
1 59
2 68
3 51

3*3
[59][0 ][0 ]
[34][34][0 ]
[0 ][0 ][51]
"""


print("###############################################")
Y_train = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
           2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
           3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
Y_pred = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
           1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
           2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
           3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
MM = metrics.confusion_matrix(Y_train,Y_pred)
print("matriz de confusion:\n%s" %(MM))
reporte = metrics.classification_report(Y_train,Y_pred)
print(reporte)

