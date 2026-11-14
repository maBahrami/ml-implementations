import gymnasium as gym
import numpy as np
from collections import defaultdict
import pandas as pd

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")

epsilon = 0.05
gamma = 1.0
alpha = 0.2
nb_actions = env.action_space.n

def argmax_rand(arr):
    # breaks ties randomly, np.argmax() always picks first max
    return(np.random.choice(np.flatnonzero(arr == np.max(arr))))

def sarsa(epsilon, alpha, ep):
    Q = defaultdict(float)

    def policy(St, pi):
        return np.random.choice([0, 1, 2, 3], p = [pi[(St, a)] for a in [0, 1, 2, 3]])
    
    pi = defaultdict(lambda: 1 / nb_actions)

    for _ in range(ep):
        s, _ = env.reset()
        a = policy(s, pi)

        while True:
            s_, r, terminated, truncated, _ = env.step(a)
            a_ = policy(s_, pi)

            Q[(s, a)] = Q[(s, a)] + alpha * (r + gamma * Q[(s_, a_)] - Q[(s, a)])
            A_star = argmax_rand(np.array([Q[(s, b)] for b in range(nb_actions)]))

            for a in range(nb_actions):
                if a == A_star:
                    pi[(s, a)] = 1 - epsilon + epsilon / nb_actions
                else:
                    pi[(s, a)] = epsilon / nb_actions

            s = s_
            a = a_

            if terminated or truncated:
                break

    return Q , pi


Q, pi = sarsa(epsilon, alpha, ep = 50000)

pi = pd.DataFrame(pi.items(), columns=["state_action", "probability"])
greedy = [max(range(nb_actions), key=lambda b: Q[(s, b)]) for s in range(16)]
print(np.array(list("LDRU"))[greedy].reshape(4, 4))