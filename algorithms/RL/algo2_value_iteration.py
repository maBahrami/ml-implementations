import gymnasium as gym
import numpy as np
import time

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")

nS = env.observation_space.n
nA = env.action_space.n
gamma = 0.99
threshold = 1e-10

initial_V = np.zeros(nS)

def value_iteration(V_prev):

    V = np.zeros(nS)

    while True:
        Q = np.zeros((nS, nA))
        for state in range(nS):
            for action in range(nA):
                for prob, next_state, reward, _ in env.unwrapped.P[state][action]:
                    Q[state, action] += prob * (reward + gamma * V_prev[next_state])

            V[state] = np.max(Q[state, :])

        if np.max(np.abs(V - V_prev)) < threshold:
            break

        V_prev = np.copy(V)

    Q = np.zeros((nS, nA))
    for state in range(nS):
        for action in range(nA):
            for prob, next_state, reward, _ in env.unwrapped.P[state][action]:
                Q[state, action] += prob * (reward + gamma * V[next_state])

    policy = np.zeros((nS, nA))
    cols = Q.argmax(axis=1)
    rows = range(nS)
    policy[rows, cols] = 1

    return policy

start = time.time()
policy = value_iteration(initial_V)
end = time.time()
elapsed_time = end - start
print(policy)
print(f"elapsed time: {elapsed_time:.2f}")