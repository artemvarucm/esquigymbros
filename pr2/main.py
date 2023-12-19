from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import gymnasium as gym
import pygame # hace falta para pintar el tablero graficamente en una ventanita
from Qlearning import *



env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=False, render_mode="none",desc = generate_random_map(size=2)) # reemplazar none por human para ver el muneco

# Apartado e)
qle = Qlearning(env.observation_space.n, 4, 0.7, 0.95, 10, env, 99,0.9)
  # semilla de numeros pseudo-aleatorios
'''Realizamos el entrenamiento con los parámetros indicados, y mostramos la matriz obtenida.'''
print(qle.entrenamiento())
