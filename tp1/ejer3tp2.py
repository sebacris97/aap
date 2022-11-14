import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('AUTOS.csv')
print(df.to_string())

scatter_plot=df.plot.scatter(x=' length',y=' price') #relaciono precio con largo 
d=scatter_plot.plot()

conteo = df[' make'].value_counts()
plt.style.use("bmh")                   
a=plt.bar(conteo.index, conteo) #grafico de barras observo marca con mas autos

b=df.boxplot(column=[' price'], by=' make', vert=False) #grafico de caja

c=df[[' price']].hist() #histograma obtengo cantida dde autos a determinado precio


plt.show()
plt.close('all')

