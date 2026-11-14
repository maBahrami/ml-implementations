import gymnasium as gym
import numpy as np
from collections import defaultdict
import pandas as pd

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")

def policy(state):
    # a deterministic policy as an example
    if state in [1, 4]:
        return 0
    elif state in [2, 9, 14]:
        return 1
    elif state in [0, 6, 13]:
        return 2
    else:
        return 3

def td_learning(env, alpha=0.2, gamma=1.0, ep=1000):
    V = defaultdict(int)

    for _ in range(ep):
        s = env.reset()[0]
        # print(s)

        while True:
            a = policy(s)
            s_, r, truncated, terminated, _ = env.step(a)
            V[s] = V[s] + alpha * ((r + gamma * V[s_]) - V[s])

            if truncated or terminated:
                break

            s = s_

    return V

V = td_learning(env)
V = pd.DataFrame(V.items(), columns=["state", "value"])
V = V.sort_values(by="state").reset_index(drop=True)
print(V)
