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

def q_learning(env, ep, eps=1.0, eps_min=0.05, decay=0.9999):
    Q = defaultdict(float)
    for i in range(ep):
        eps = max(eps_min, eps * decay)

        s, _ = env.reset()
        
        while True:
            a = policy(s, Q, eps)
            s_, r, terminated, truncated, _ = env.step(a)
            q_max = max([Q[(s_, at)] for at in range(nb_actions)])
            Q[(s, a)] = Q[(s, a)] + alpha*(r + gamma*q_max - Q[(s, a)])

            s = s_

            if terminated or truncated:
                break

    return Q

Q = q_learning(env, ep=150000)

greedy = [max(range(nb_actions), key=lambda b: Q[(s, b)]) for s in range(16)]
print(np.array(list("LDRU"))[greedy].reshape(4, 4))

Q_df = pd.DataFrame(sorted(Q.items()), columns=["state-action", "q"])
print(Q_df)