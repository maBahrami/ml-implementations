import gymnasium as gym
import numpy as np
from collections import defaultdict
import pandas as pd

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")

gamma = 1.0
alpha = 0.2
nb_actions = env.action_space.n

def argmax_rand(arr):
    # breaks ties randomly, np.argmax() always picks first max
    return(np.random.choice(np.flatnonzero(arr == np.max(arr))))

def policy(st, Q, eps):
    if np.random.uniform() < eps:
        return env.action_space.sample()
    else:
        return argmax_rand(np.array([Q[(st, at)] for at in range(nb_actions)]))

def sarsa(env, ep, eps=1.0, eps_min=0.05, decay=0.9999):
    Q = defaultdict(float)
    for i in range(ep):
        eps = max(eps_min, eps * decay)

        s, _ = env.reset()
        a = policy(s, Q, eps)

        while True:
            s_, r, terminated, truncated, _ = env.step(a)
            a_ = policy(s_, Q, eps)

            Q[(s, a)] = Q[(s, a)] + alpha*(r + gamma*Q[(s_, a_)] - Q[(s, a)])

            s = s_
            a = a_

            if terminated or truncated:
                break

    return Q

Q = sarsa(env, ep=150000)
Q = pd.DataFrame(Q.items(), columns=["state-action", "q"])
print(Q.head(32))
print(Q.tail(32))