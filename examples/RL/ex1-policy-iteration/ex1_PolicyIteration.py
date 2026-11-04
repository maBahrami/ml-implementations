import gymnasium as gym
import numpy as np
import time

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")

nS = env.observation_space.n
nA = env.action_space.n
gamma = 0.99
threshold = 1e-10


def policy_evaluation(policy):
    iteration = 0
    prev_V = np.zeros(nS)

    while True:
        iteration +=1 
        delta = 0
        V = np.zeros(nS)

        for state in range(nS):
            for action in range (nA):
                for prob, next_state, reward, _ in env.unwrapped.P[state][action]:
                    V[state] += policy[state][action] * prob * (reward + gamma*prev_V[next_state])

        delta = np.max(np.abs(V - prev_V))
        if delta <= threshold:
            break

        prev_V = np.copy(V)

    return V, iteration


def policy_improvement(V):
    policy = np.zeros((nS, nA))
    q = np.zeros((nS, nA))

    for state in range(nS):
        for action in range(nA):
            for prob, next_state, reward, _ in env.unwrapped.P[state][action]:
                q[state][action] += prob * (reward + gamma*V[next_state])

    cols = np.argmax(q, axis=1)
    rows = np.arange(nS)
    policy[rows, cols] = 1

    return policy


def policy_iteration(policy):
    i = 0
    while True:
        i += 1
        V, _ = policy_evaluation(policy)
        policy_updated = policy_improvement(V)

        if np.array_equal(policy, policy_updated):
            break

        policy = policy_updated

    return policy, i

    

random_policy = np.ones([nS, nA]) / nA
#print(f"\nRandom Policy:\n\n {random_policy}\n")

# deterministic policy example of the book
book_policy = np.array([
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

start = time.time()     
optimal_policy_1, i_1 = policy_iteration(random_policy)
end = time.time()
elapsed_time_1 = end - start
print(f"optimal policy is:\n\n{optimal_policy_1}")
print(f"\nnumber of policy iteration: {i_1}\n")
print(f"elapsed time: {elapsed_time_1:.3f}s")

start = time.time()     
optimal_policy_2, i_2 = policy_iteration(book_policy)
end = time.time()
elapsed_time_2 = end - start
print(f"optimal policy is:\n\n{optimal_policy_2}")
print(f"\nnumber of policy iteration: {i_2}\n")
print(f"elapsed time: {elapsed_time_2:.3f}s")

if np.array_equal(optimal_policy_1, optimal_policy_2):
    print("\n** both policies are the same **\n")