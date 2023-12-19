import gymnasium as gym
import pygame # hace falta para pintar el tablero graficamente en una ventanita
from Qlearning import *

random.seed(10)  # semilla de numeros pseudo-aleatorios

env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=False, render_mode="none") # reemplazar none por human para ver el muneco


# Apartado e)
qle = Qlearning(env.observation_space.n, 4, 0.7, 0.95, 15000, env, 99,0.3)

'''Realizamos el entrenamiento con los parámetros indicados, y mostramos la matriz obtenida.'''
print(qle.entrenamiento())
