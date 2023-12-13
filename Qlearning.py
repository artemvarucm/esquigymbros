import numpy as np
import random
class Qlearning:
    def __init__(self, n, m, alpha, gamma, n_episodios, env, max_steps):
        # n - numero de estados
        # m - numero de acciones posibles
        self.matriz_q = self.inicializacion_q(n, m)
        self.alpha = alpha
        self.gamma = gamma
        self.n_episodios = n_episodios
        self.env = env
        self.max_steps = max_steps
        # Configuracion interna
        random.seed(10) # semilla de numeros pseudo-aleatorios
        self.epsilon = 0.3


    def inicializacion_q(self,n,m):
        return np.zeros((n,m))

    def accion_maxima_siguiente(self, estado):
        num_aleatorio = random.random() # numero aleatorio entre 0 y 1
        if (num_aleatorio <= self.epsilon):
            accion = self.hacer_exploracion(estado)
        else:
           accion = self.hacer_explotacion(estado)

        return accion

    def hacer_explotacion(self, estado):
        # Acción que maximiza Q en el estado actual (siempre posible de realizar)
        posibles =  self.acciones_posibles(estado)
        fila_q_posibles = self.matriz_q[estado, posibles]
        return posibles[np.argmax(fila_q_posibles)]  # primer max de la fila que corresponde al estado.

    def hacer_exploracion(self, estado):
        # Accion aleatoria (siempre posible de realizar)
        return random.choice(self.acciones_posibles(estado))

    def acciones_posibles(self, estado):
        # FIXME Se puede optimizar usando la libreria gym
        # Devuelve acciones posibles para el estado
        # Hallamos coordenadas
        col = estado % 4
        fila = estado // 4 # div. entera
        #Dimension del tablero, o, ultimo indice del tablero cuadrado
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

    def resetear_entorno(self):
        return self.env.reset()[0]  # env.env fixme

    def aplicar_accion(self, accion):
        # Devuelve los tres primeros elementos de la tupla
        return self.env.step(accion)[:3]

    def calcular_matriz_q(self, s, s_next, accion, recompensa, alpha, gamma):
        # FIXME tuve que anadir estado_sig
        matriz_q_nueva = self.matriz_q.copy()
        max_q_next = max(matriz_q_nueva[s_next, self.acciones_posibles(s_next)])
        matriz_q_nueva[s, accion] = (1 - alpha) * matriz_q_nueva[s, accion] + alpha * (recompensa + gamma * max_q_next)
        return matriz_q_nueva

    def entrenamiento(self,max_steps):
        for episodio in range(self.n_episodios):
            estado = self.resetear_entorno()
            step = 0
            acabar = False
            while step <= max_steps and not acabar:
                accion = self.accion_maxima_siguiente(estado)
                estado_sig, recompensa, final = self.aplicar_accion(accion)
                self.matriz_q = self.calcular_matriz_q(estado, estado_sig, accion, recompensa, self.alpha, self.gamma)
                step = step + 1
                if final:
                    acabar = final
                else:
                    estado = estado_sig

        return self.matriz_q
        '''
    Para cada episodio hasta n_episodios hacer
        estado <- resetear_entorno()
        step <- 0
        Para cada step hasta un máximo de max_steps hacer
            accion <- obtener_accion_siguiente(estado)
            estado_sig, recompensa <- aplicar_accion(acción)
            matriz_q <- calcular_matriz_q(estado, acción, recompensa, alpha, gamma)
        si final(estado_sig) entonces salir
        estado <- estado_sig
    devolver matriz_q

        '''

    def visualizarCamino(self):
        estado = self.resetear_entorno()
        while (1 == 1):
            accion = self.hacer_explotacion(estado)
            estado_sig, recompensa, final = self.aplicar_accion(accion)
            if (final):
                break
            estado = estado_sig


# env.step(accion)
# sig
# recompensa
# final
# info
