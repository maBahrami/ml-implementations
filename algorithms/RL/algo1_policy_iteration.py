import gymnasium as gym
import numpy as np
import time

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")

nS = env.observation_space.n
nA = env.action_space.n
gamma = 0.99
threshold = 1e-10

# initial policies
initial_policy_1 = np.ones((nS, nA)) / nA           # random policy
initial_policy_2 = np.array([                       # deterministic policy
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



def policy_evaluation(policy):

    prev_V = np.zeros(nS)
    iteration = 0

    while True:
        delta = 0
        V = np.zeros(nS)
        iteration += 1

        for state in range(nS):
            for action in range(nA):
                for prob, next_state, reward, _ in env.unwrapped.P[state][action]:      # env.unwrapped.P is a dictionary
                    V[state] += policy[state, action] * prob * (reward + gamma * prev_V[next_state])

        delta = np.max(np.abs(V-prev_V))

        if delta <= threshold:
            break

        prev_V = np.copy(V)
        

    return V, iteration


def policy_improvement(V):

    Q = np.zeros((nS, nA))

    for state in range(nS):
        for action in range(nA):
            for prob, next_state, reward, _ in env.unwrapped.P[state][action]:
                Q[state, action] += prob * (reward + gamma * V[next_state])

    policy = np.zeros((nS, nA))
    cols = Q.argmax(axis=1)
    rows = range(nS)
    policy[rows, cols] = 1

    return Q, policy


def policy_iteration(policy):

    i = 0
    Q_prev = np.zeros((nS, nA))

    while True:
        i += 1
        V, _ = policy_evaluation(policy)
        Q, policy_updated = policy_improvement(V)
        
        if np.array_equal(policy, policy_updated):
            break

        policy = policy_updated

    return policy, i


# finding optimal policy stating from a random initial policy
start = time.time()     
optimal_policy_1, iteration_1 = policy_iteration(initial_policy_1)
end = time.time()
elapsed_time_1 = end - start
print(f"\noptimal policy is:\n\n{optimal_policy_1}")
print(f"\nnumber of policy iteration: {iteration_1}")
print(f"elapsed time: {elapsed_time_1:.3f}s")

# finding optimal policy stating from a deterministic initial policy
start = time.time()     
optimal_policy_2, iteration_2 = policy_iteration(initial_policy_2)
end = time.time()
elapsed_time_2 = end - start
print("--------------------------------------------")
print(f"\noptimal policy is:\n\n{optimal_policy_2}")
print(f"\nnumber of policy iteration: {iteration_2}")
print(f"elapsed time: {elapsed_time_2:.3f}s")

# checking the final optimal policies resulting from two different initial policies
print("--------------------------------------------")
if np.array_equal(optimal_policy_1, optimal_policy_2):
    print("** both policies are the SAME **")
else:
    print("** DIFFERENT optimal policies **")
print("--------------------------------------------")