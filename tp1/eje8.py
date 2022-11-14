import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

"""
df = pd.read_csv('AUTOS.csv')
#df.dropna(inplace=True)
print(df)

print('\n')
etiq = ['chico','grande']
columna = pd.cut(df[' engine-size'], bins=len(etiq), labels=etiq)
print(columna)
df[' engine-size2'] = pd.Series.to_frame(columna)
print(df[[' engine-size',' engine-size2']].to_string())

print(pd.value_counts(df[' engine-size2']))

#(60.735 , 193.5]
#(19.,5 , 326.0]


"""

df = pd.read_csv('AUTOS.csv')
#df.dropna(inplace=True)
print(df)
print('\n')
etiq = ['muy bajo','bajo','normal','alto','muy alto']
columna = pd.cut(df[' height'], bins=len(etiq), labels=etiq)
print(columna)
df[' height2'] = pd.Series.to_frame(columna)
print(df[[' height',' height2']].to_string())
print(pd.value_counts(df[' height2']))
conteo = df[' height2'].value_counts()
plt.style.use("bmh")                   
a=plt.bar(conteo.index, conteo)
plt.show()
