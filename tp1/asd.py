import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
matriz=np.random.randint(100, size=(100))
df = pd.DataFrame(matriz,columns=['A'])
print(df)
print('mediana: ',df.median())
print('media: ',df.mean())
print('moda: ',pd.Series(df.values.flatten()).mode()[0])

#elimino 2 mas bajos y 2 mas altos
matriz=matriz[2:-3]
df = pd.DataFrame(matriz,columns=['A'])
print('mediana: ',df.median())
print('media: ',df.mean())
print('moda: ',pd.Series(df.values.flatten()).mode()[0])
"""

"""
matriz=[15]*100
df = pd.DataFrame(matriz,columns=['A'])
print(df)
print('mediana: ',df.median())
print('media: ',df.mean())
print('moda: ',pd.Series(df.values.flatten()).mode()[0])
"""

COLOR=[200, 250, 252, 10, 189, 211, 0, 245]
df = pd.DataFrame(COLOR,columns=['A'])
print('mediana: ',df.median())
print('media: ',df.mean())
print('moda: ',pd.Series(df.values.flatten()).mode()[0])
df.plot(kind='box',vert=False);
plt.show()
