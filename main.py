import gym
import pygame # hace falta para pintar el tablero graficamente en una ventanita
from Qlearning import *

env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=False, render_mode="none") # reemplazar none por human para ver el muneco
env.action_space.seed(12)

# Apartado e)
qle = Qlearning(env.observation_space.n, 4, 0.7, 0.95, 15000, env, 99)

# Matriz Q

print(qle.entrenamiento(99))
# Para ver la solucion
qle.env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=False, render_mode="human")
while (1==1):
    qle.visualizarCamino()

'''
print("Tamaño de espacio de estados", env.observation_space)
print("Estado aleatorio", env.observation_space.sample())

size_estados = env.observation_space.n
print("Hay", size_estados, " estados posibles.")

env.env.reset()
#env.env.render()

print("Acciones posibles", env.action_space)
print("Acción aleatoria", env.action_space.sample())
size_acciones = env.action_space.n
print("Hay", size_acciones, " acciones posibles.")
'''