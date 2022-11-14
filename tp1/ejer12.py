import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

matriz = np.array([[1.65, 75], [1.81, 86], [1.70, 82],
                   [1.62, 78], [1.74, 77], [1.70, 87],
                   [1.80, 90], [1.73, 83], [1.68, 80]])

df = pd.DataFrame(matriz,columns=['altura','peso'])

mini = df['peso'].min()
maxi = df['peso'].max()
df['pesolineal'] = (df['peso']-mini)/(maxi-mini)


media = df['peso'].mean()
desvio = df['peso'].std()
df['pesonorm'] = (df['peso']-media)/desvio

print(df['peso']+1)


Q = df['peso'].quantile([0.25, 0.5, 0.75]).values
df['robust'] = (df['peso']-Q[0]) / (Q[2]-Q[0])


print(df)

a=df['peso'].plot(kind='box',vert=False);
plt.show()
plt.close()

b=df[['pesolineal','pesonorm','robust']].plot(kind='box',vert=False);
plt.show()
plt.close()

plt.close()
