import numpy as np
class Qlearning:
    def __init__(self, n, m, alpha, gamma, n_episodios, env,
                 max_steps):
        self.matriz_q = self.inicializacion_q(n, m)
        self.alpha = alpha
        self.gamma = gamma
        self.n_episodios = n_episodios
        self.env = env
        self.max_steps = max_steps


    def inicializacion_q(self,n,m):
        return np.zeros((n,m))

    def accion_maxima_siguiente(self, estado):
        return np.argmax(self.matriz_q[estado]) #primer max de la fila que corresponde al estado.


    def entrenamiento(self,max_steps):
        for episodio in self.n_episodios:
            estado = self.env.reset() # env.env fixme
            step = 0
            acabar = False
            while step <= max_steps and not acabar:
                accion = self.accion_maxima_siguiente(estado)
                #estado_sig,recompensa = #aplicar_accion
                #matriz_q < - calcular_matriz_q(estado, acción, recompensa, alpha, gamma)
                step = step + 1
                if final(estado_sig):
                    acabar = True
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


    def final(self,estado):
        return self.env.step()


# env.step(accion)
# sig
# recompensa
# final
# info