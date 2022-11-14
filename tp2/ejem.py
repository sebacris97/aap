import numpy as np
from grafica import *

class Perceptron():
    
    def __init__(self, alpha=0.01, n_iter=50, draw=0, title=['X1','X2'],show_ite=False, random_state=None):
        self.alpha = alpha
        self.n_iter = n_iter
        self.draw = draw
        self.title = title
        self.show_ite = show_ite
        self.random_state = random_state

    def fit(self, entradas, salida):   
        nCantEjemplos = entradas.shape[0] #n filas
        nAtrib = entradas.shape[1] #n columnas
        #--- Pesos iniciales ---
        rgen = np.random.RandomState(self.random_state)
        self.W_ = rgen.uniform(-0.5, 0.5, size=nAtrib) 
        self.b_ = rgen.uniform(-0.5, 0.5)
        if self.draw:
            ph=dibuPtosRecta(entradas,salida, self.W_, self.b_, titulos = self.title)
        #--- Parametros del PERCEPTRON ---
        MAX_ITE = 100
        alfa = 0.1
        self.ite_ = 0
        # --- Entrenamiento del PERCEPTRON ---
        hubo_cambio=True
        while hubo_cambio and self.ite_<MAX_ITE:
            hubo_cambio=False
            for e in range(nCantEjemplos):
                neta = self.b_
                for a in range(nAtrib):
                    neta  +=  self.W_[a]*entradas[e,a]
                y = 1*(neta>0)
                if y!=salida[e]:
                    hubo_cambio=True
                    for a in range(nAtrib):
                        self.W_[a] = self.W_[a]+alfa*(salida[e]-y)*entradas[e,a]
                    self.b_ = self.b_+alfa*(salida[e]-y)*1
            if self.draw:
                ph = dibuPtosRecta(entradas,salida, self.W_, self.b_,ph=ph)
            self.ite_ += 1
        if self.show_ite:
            print('iteraciones: %d' %self.ite_)
