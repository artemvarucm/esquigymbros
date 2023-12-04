import gym
import pygame
import numpy as np

env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=False)

print("Tamaño de espacio de estados", env.observation_space)
print("Estado aleatorio", env.observation_space.sample())

size_estados = env.observation_space.n
print("Hay", size_estados, " estados posibles.")

env.env.reset()
env.env.render()

print("Acciones posibles", env.action_space.n)
print("Acción aleatoria", env.action_space.sample())
size_acciones = env.action_space.n
print("Hay", size_acciones, " acciones posibles.")
