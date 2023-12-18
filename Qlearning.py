import numpy as np
import random
class Qlearning:
    def __init__(self, n, m, alpha, gamma, n_episodios, env, max_steps,epsilon):
        # n - numero de estados
        # m - numero de acciones posibles
        self.matriz_q = self.inicializacion_q(n, m)
        self.alpha = alpha
        self.gamma = gamma
        self.n_episodios = n_episodios
        self.env = env
        self.max_steps = max_steps
        # Configuracion interna
        self.epsilon = epsilon


    def inicializacion_q(self,n,m):
        '''Crea la matriz Q de ceros dadas las dimensiones n x m'''
        return np.zeros((n,m))


    def accion_maxima_siguiente(self, estado):
        '''Elige un numero aleatorio entre 0 y 1. Si el numero es menor o igual que epsilon
        exploraremos y si es mayor, explotaremos.
        Devuelve la accion elegida.'''
        num_aleatorio = random.uniform(0,1) # numero aleatorio entre 0 y 1
        if (num_aleatorio <= self.epsilon):
            accion = self.hacer_exploracion(estado)
        else:
           accion = self.hacer_explotacion(estado)

        return accion

    def hacer_explotacion(self, estado):
        ''' Acción que maximiza Q en el estado actual (siempre posible de realizar)'''
        posibles =  self.acciones_posibles(estado)
        fila_q_posibles = self.matriz_q[estado, posibles]
        return posibles[np.argmax(fila_q_posibles)]  # primer max de la fila que corresponde al estado.

    def hacer_exploracion(self, estado):
        ''' Accion aleatoria (siempre posible de realizar)'''
        return random.choice(self.acciones_posibles(estado))

    def acciones_posibles(self, estado):
        '''Devuelve acciones posibles para el estado dado'''

        # Hallamos coordenadas
        col = estado % 4
        fila = estado // 4 # div. entera
        acciones = list()
        dim = np.sqrt(self.matriz_q.shape[0]) - 1
        if (col != 0): # podemos ir a la izquierda
            acciones.append(0)
        if (fila != dim): # podemos ir abajo
            acciones.append(1)
        if (col != dim): # podemos ir a la derecha
            acciones.append(2)
        if (fila != 0): # podemos ir arriba
            acciones.append(3)

        return acciones

    def resetear_entorno(self):
        return self.env.reset()[0]

    def aplicar_accion(self, accion):
        '''Devuelve los tres primeros elementos de la tupla'''
        return self.env.step(accion)[:3]

    def calcular_matriz_q(self, s, s_next, accion, recompensa, alpha, gamma):
        '''Actualiza la matriz Q segun los parámetros proporcionados.'''
        max_q_next = max(self.matriz_q[s_next, self.acciones_posibles(s_next)])
        self.matriz_q[s, accion] = (1 - alpha) * self.matriz_q[s, accion] + alpha * (recompensa + gamma * max_q_next)
        #return matriz_q_nueva

    def entrenamiento(self):
        '''Construimos la matriz Q mediante aprendizaje por refuerzo durante un numero de episodios
        y poniendo un limite de pasos para cada episodio.
        '''
        for episodio in range(self.n_episodios):
            estado = self.resetear_entorno()
            step = 0
            acabar = False
            while step <= self.max_steps and not acabar:
                accion = self.accion_maxima_siguiente(estado)
                estado_sig, recompensa, final = self.aplicar_accion(accion)
                self.calcular_matriz_q(estado, estado_sig, accion, recompensa, self.alpha, self.gamma)
                step = step + 1
                if final:
                    acabar = True
                else:
                    estado = estado_sig

        return self.matriz_q





