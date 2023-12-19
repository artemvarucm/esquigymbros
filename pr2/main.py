from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import gymnasium as gym
import pygame # hace falta para pintar el tablero graficamente en una ventanita
from Qlearning import *

# Ejercicio 3


# a) suelo resbaladizo
env_a = gym.make("FrozenLake-v1",render_mode="none",map_name="4x4",is_slippery=True)

qle_a = Qlearning(env_a.observation_space.n, 4, 0.7, 0.95, 15000, env_a, 99,0.3)

print(qle_a.entrenamiento())

input("Enter para seguir...")
# Aumentamos n de episodios
qle_a = Qlearning(env_a.observation_space.n, 4, 0.7, 0.95, 30000, env_a, 99,0.3)

print(qle_a.entrenamiento())

input("Enter para seguir...")
# b) configuraciones especificas

# PRIMERA
descT1 = [
    "SFFH",
    "HHFH",
    "FFFH",
    "FFFG"
]
env_b_t1 = gym.make("FrozenLake-v1",render_mode="none",map_name="4x4",is_slippery=False, desc = descT1)

qle_b_t1 = Qlearning(env_b_t1.observation_space.n, 4, 0.7, 0.95, 30000, env_b_t1, 99,0.3)

print(qle_b_t1.entrenamiento())

input("Enter para seguir...")
# Aumentamos n de episodios

qle_b_t1 = Qlearning(env_b_t1.observation_space.n, 4, 0.7, 0.95, 30000, env_b_t1, 99,0.3)

print(qle_b_t1.entrenamiento())
input("Enter para seguir...")

# SEGUNDA
descT2 = [
    "SHFF",
    "FFFF",
    "FFHF",
    "HFHG"
]

env_b_t2 = gym.make("FrozenLake-v1",render_mode="none",map_name="4x4",is_slippery=False, desc = descT2)

qle_b_t2 = Qlearning(env_b_t2.observation_space.n, 4, 0.7, 0.95, 30000, env_b_t2, 99,0.3)

print(qle_b_t2.entrenamiento())
input("Enter para seguir...")

# Aumentamos n de episodios

qle_b_t2 = Qlearning(env_b_t2.observation_space.n, 4, 0.7, 0.95, 30000, env_b_t2, 99,0.3)

print(qle_b_t2.entrenamiento())

input("Enter para seguir...")

# c) random life - random rules
justRandomMap = generate_random_map(size=4)
# Mostramos el mapa generado
for line in justRandomMap:
    print(line)

env_c = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=False, render_mode="none",desc = justRandomMap)

qle_c = Qlearning(env_c.observation_space.n, 4, 0.7, 0.95, 30000, env_c, 99,0.3)

print(qle_c.entrenamiento())