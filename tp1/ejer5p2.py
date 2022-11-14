import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

df = pd.read_csv('AUTOS.csv')
dfgas = df[df[' fuel-type']=='gas']
dfdiesel = df[df[' fuel-type']=='diesel']
print(dfgas)
print(dfdiesel)
fig=df.boxplot(column=[' price'], by=' fuel-type', vert=False)
#para ver con el mouse bien
#fig.get_xaxis().set_major_formatter(plt.LogFormatter(10,  labelOnlyBase=False))
#fig.get_yaxis().set_major_formatter(plt.LogFormatter(10,  labelOnlyBase=False))

columndiesel = dfdiesel[' price']
columngas = dfgas[' price']
MEDIANA_DIESEL = columndiesel.median()
MEDIANA_GAS = columngas.median()
print('mediana disel: ',MEDIANA_DIESEL)
print('mediana gas: ',MEDIANA_GAS)
Qd = columndiesel.quantile([0.25, 0.5, 0.75]).values
print('quartiles diesel',Qd)
Qg = columngas.quantile([0.25, 0.5, 0.75]).values
print('quartiles gas: ',Qg)
max_valueD = columndiesel.max()
min_valueD = columndiesel.min()
print('maximo diesel:',max_valueD)
print('minimo diesel:',min_valueD)
max_valueG = columngas.max()
min_valueG = columngas.min()
print('maximo gas:',max_valueG)
print('minimo gas:',min_valueG)
RICD=Qd[2]-Qd[0]
RICG=Qg[2]-Qg[0]
print('ric diesel: ',RICD)
print('ric gas: ',RICG)

plt.show()

mini = df[' symboling'].min()
maxi = df[' symboling'].max()
df[' symboling']= (df[' symboling']-mini)/(maxi-mini)
print(df[' symboling'].to_string())
