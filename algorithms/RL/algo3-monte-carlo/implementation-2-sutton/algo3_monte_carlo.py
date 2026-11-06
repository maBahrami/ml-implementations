import numpy as np
import pandas as pd
from collections import defaultdict
import gymnasium as gym


env = gym.make("Blackjack-v1", natural=False, sab=False)
num_episode = 100000
gamma = 1


def policy_1(state):
    """considering any initial policy out of nowhere"""
    return 0 if state > 19 else 1

def generate_episode(env, policy):
    done = True
    trajectory = []

    while True:
        if done:
            (St, _), Rt, done = env.reset(), None, False
        else:
            St, Rt, done, _, _ = env.step(At)
        At = policy(St[0])
        trajectory.append((St, Rt, done, At))
        if done:
            break
    return trajectory, len(trajectory)-1    # index of terminal state

def predict(policy, num_episode):
    V = defaultdict()
    Returns = defaultdict(list)

    for _ in range(num_episode):
        traj, T = generate_episode(env, policy)
        G = 0

        # staring form second-to-last state till the first one
        for t in range(T-1, -1, -1):
            St, _, _, _ = traj[t]
            _, Rt_1, _, _ = traj[t+1]
            G = gamma * G + Rt_1

            if St not in [traj[i][0] for i in range(0, t)]:
                Returns[St].append(G)
                V[St] = np.average(Returns[St])

    return V


# tra, a = generate_episode(env, policy=policy_1)
# print("\n", tra, "\n")
# print(a)

V = predict(policy_1, num_episode = 2)
