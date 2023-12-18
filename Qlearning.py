import numpy as np
import random
class Qlearning:
    def __init__(self, n, m, alpha, gamma, n_episodios, env, max_steps, epsilon):
        self.matriz_q = self.inicializacion_q(n, m)
        self.alpha = alpha
        self.gamma = gamma
        self.n_episodios = n_episodios
        self.env = env
        self.max_steps = max_steps
        self.epsilon = epsilon

    def inicializacion_q(self,n,m):
        return np.zeros((n,m))

    def hacer_exploracion(self,estado):
        return random.choice(self.acciones_posibles(estado))

    def hacer_explotacion(self,estado):
        posibles = self.acciones_posibles(estado)
        fila_q_posibles = self.matriz_q[estado, posibles]
        return posibles[np.argmax(fila_q_posibles)]  # primer max de la fila que corresponde al estado.
    def aplicar_accion(self,accion):
        return self.env.step(accion)[:3]

    def calcular_matriz_q(self,estado, estado_sig, accion,recompensa,alpha,gamma):
        max_q_next = np.max(self.matriz_q[estado_sig][self.acciones_posibles(estado_sig)])
        self.matriz_q[estado][accion] = (1-alpha)* self.matriz_q[estado][accion] + alpha*(recompensa + gamma * max_q_next)

    def accion_maxima_siguiente(self, estado):
        numero_aleatorio = random.uniform(0, 1)
        if numero_aleatorio <= self.epsilon:
            accion = self.hacer_exploracion(estado)
        else:
            accion = self.hacer_explotacion(estado)
        return accion

    def resetear_entorno(self):
        return self.env.reset()[0]

    def entrenamiento(self):
        for episodio in range(self.n_episodios):
            estado = self.resetear_entorno()
            step = 0
            acabar = False
            while step <= self.max_steps and not acabar:
                accion = self.accion_maxima_siguiente(estado)
                estado_sig,recompensa,final = self.aplicar_accion(accion)
                self.calcular_matriz_q(estado,estado_sig, accion, recompensa, self.alpha, self.gamma)
                step = step + 1
                if final:
                    acabar = True
                else:
                    estado = estado_sig

        return self.matriz_q

    def acciones_posibles(self, estado):
        col = estado % 4
        fila = estado // 4  # div. entera
        # Dimension del tablero, o, ultimo indice del tablero cuadrado
        dim = np.sqrt((self.matriz_q.shape[0])) - 1
        acciones = list()
        if (col != 0):  # podemos ir a la izquierda
            acciones.append(0)
        if (fila != dim):  # podemos ir abajo
            acciones.append(1)
        if (col != dim):  # podemos ir a la derecha
            acciones.append(2)
        if (fila != 0):  # podemos ir arriba
            acciones.append(3)
        return acciones
