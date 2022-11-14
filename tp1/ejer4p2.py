import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('AUTOS.csv')
df.dropna(inplace=True)
df['horsepower'].plot(kind='box',vert=False);
plt.show()

column = df["horsepower"]
MEDIANA = column.median()
print(MEDIANA)
Q = column.quantile([0.25, 0.5, 0.75]).values
print(Q)
max_value = column.max()
min_value = column.min()
print(max_value)
print(min_value)
RIC=Q[2]-Q[0]
print('RIC = ',RIC)

"""
ej4:
Q1 69
Q2 o MEDIANA 88
Q3 114	
MAX 200
MIN 48
bigote inferior 48
bigote superior 176
RIC = 45
atipicos leves = 1,5 * RIC =1,5*45 = 67,5
atipicos extermos 3 * RIC = =3*45 = 135
lim inf = q1 - atipicos leves = 69 + 67,5 = 136,5
lim sup = q3 + atipicos leves = 114 + 67,5 = 181,5
"""
