import gymnasium as gym
import numpy as np
import time

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")

nS = env.observation_space.n
nA = env.action_space.n
gamma = 0.99
threshold = 1e-10



def policy_evaluation(policy):

    prev_V = np.zeros(nS)
    iteration = 0

    while True:
        delta = 0
        V = np.zeros(nS)
        iteration += 1

        for state in range(nS):
            for action in range(nA):
                for prob, next_state, reward, _ in env.unwrapped.P[state][action]:
                    V[state] += policy[state][action] * prob * (reward + gamma * prev_V[next_state])

        delta = np.max(np.abs(V-prev_V))

        if delta <= threshold:
            break

        prev_V = np.copy(V)
        

    return V, iteration




policy_1 = np.ones((nS, nA)) / nA           # random policy
policy_2 = np.array([                       # deterministic policy
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                    ])


V, iteration = policy_evaluation(policy_2)
print("\n", V.reshape((4, 4)), "\n\nnumber of iterations: ",iteration, "\n")