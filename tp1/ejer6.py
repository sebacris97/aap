import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('AUTOS.csv')
#df.dropna(inplace=True)
df = df[[' curb-weight', ' highway-mpg']]
#relaciono curb weight y highway mpg
scatter_plot=df.plot.scatter(x=' curb-weight',y=' highway-mpg')
d=scatter_plot.plot()
print(df.corr()) #coeficiente de correlacion lineal
print(df.cov()) #covarianza
plt.show()
