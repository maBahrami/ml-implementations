import gymnasium as gym
import numpy as np
import time

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")

nS = env.observation_space.n
nA = env.action_space.n
gamma = 0.99
threshold = 1e-10

policy = np.ones((nS, nA)) / nA
# print(random_policy)

def policy_evaluation():

    prev_V = np.zeros(nS)
    iteration = 0

    while True:
        delta = 0
        V = np.zeros(nS)

        for state in range(nS):
            for action in range(nA):
                for prob, next_state, reward, _ in env.unwrapped.P[state][action]:
                    V[state] += policy[state][action] * prob * (reward + gamma * prev_V[next_state])

        delta = np.max(np.abs(V-prev_V))

        if delta <= threshold:
            break

        prev_V = np.copy(V)
        iteration += 1

    return V, iteration


V, iteration = policy_evaluation()
print(V, "\n\nnumber of iterations: ",iteration, "\n")