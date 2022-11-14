import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('whisky.csv',encoding='ISO-8859-1')
#df.dropna(inplace=True)

#arreglo el tema de que hay algunos lujo con mayuscula y otros sin
cat = pd.value_counts(df['Categoria']).index
LIST = {}
for i in cat.to_list():
    LIST[i]=i[0].upper()+i[1:].lower()
df['Categoria'] = df['Categoria'].replace(LIST.keys(),LIST.values())

print(df)

cat = list(set(LIST.values()))
dic = {}
medias = {}
for i in cat:
    dic[i] = df.loc[df['Categoria'] == i]
    medias[i] = dic[i]['Malta'].median()
#print(dic)
#print(medias) #{'Estandar': 40.0, 'Lujo': 30.0, 'Pura_malta': 100.0}

#llena los NaN de  Malta con la media de cada categoria
df['Malta'].fillna(df.groupby(['Categoria'])['Malta'].transform('mean'),inplace=True)
df['Precio'].fillna(df.groupby(['Categoria'])['Precio'].transform('mean'),inplace=True)


#configuro los rangos de lujo,estandar, pura_malta
rangos={}
for i in dic.keys():
    rangos[i]=[dic[i]['Malta'].min(), dic[i]['Malta'].max()]
print(rangos)


for i in range(len(list(df[['Malta','Categoria']].index.values))):
    obj = df.iloc[i]
    if obj['Categoria'].__class__ == float:
        for k in list(rangos.keys()):
            if (rangos[k][0] <= obj['Malta'] and obj['Malta'] <= rangos[k][1]):
                df.at[i,'Categoria'] = k

print(df)
print(df.corr())



column = df["Precio"]
Q = column.quantile([0.25, 0.5, 0.75]).values
print(Q)
max_value = column.max()
min_value = column.min()
print(max_value,min_value)

a=df['Precio'].plot(kind='box',vert=False);
conteo = df['Precio'].value_counts()
plt.style.use("bmh")
b=plt.bar(conteo.index, conteo)
print(df['Precio'].mode())
plt.show()


