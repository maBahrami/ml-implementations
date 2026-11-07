import numpy as np
import pandas as pd
from collections import defaultdict
import gymnasium as gym
from visualization import plot_blackjack
import matplotlib.pyplot as plt


env = gym.make("Blackjack-v1", natural=False, sab=False)
num_episode = 100000
gamma = 1
nb_actions = 2

def argmax_rand(arr):
    # breaks ties randomly, np.argmax() always picks first max
    return(np.random.choice(np.flatnonzero(arr == np.max(arr))))

def generate_episode(env, policy, pi):
    done = True
    trajectory = []

    while True:
        if done:
            (St, _), Rt, done = env.reset(), None, False
        else:
            St, Rt, done, _, _ = env.step(At)

        At = policy(St, pi) if not done else None
        trajectory.append((St, Rt, done, At))
        if done:
            break

    return trajectory, len(trajectory)-1    # index of terminal state

def on_policy_mc_control(env, ep, gamma, eps):

    def policy(St, pi):
        return np.random.choice([0, 1], p=[pi[St, a] for a in range(nb_actions)])
    
    pi = defaultdict(lambda: 1 / nb_actions)
    Q = defaultdict(float)
    Returns = defaultdict(list)

    for counter in range(ep):
        traj, T = generate_episode(env, policy, pi)
        G = 0

        for t in range(T-1, -1, -1):
            _, Rt_1, _, _ =  traj[t+1]
            St, _, _, At = traj[t]
            G = gamma*G + Rt_1

            # first-visit
            if (St, At) not in [(traj[i][0], traj[i][3]) for i in range(0, t)]:
                Returns[(St, At)].append(G)
                Q[(St, int(At))] = float(np.average(Returns[(St, At)]))
                A_star = argmax_rand([Q[(St, a)] for a in range(nb_actions)])
                for a in range(nb_actions):
                    if a == A_star:
                        pi[(St, a)] = 1 - eps + eps / nb_actions
                    else:
                        pi[(St, a)] = eps / nb_actions

        if counter % 10000 == 0:
            print(f"[INFO] {counter} episodes passes")

    return Q, pi


Q, pi = on_policy_mc_control(env, 500000, gamma=1.0, eps=0.05)

# policy as a table
pi_df = pd.DataFrame(pi.items(), columns=["state-action", "probability"])
print(pi_df.head(10))
print(pi_df.tail(10))
 
# state values: V(s) = value of the best action in s
states = {s for (s, a) in Q}
V = {s: max(Q[(s, a)] for a in range(nb_actions)) for s in states}
 
# greedy policy: 0 = stick, 1 = hit
greedy = {s: int(np.argmax([Q[(s, a)] for a in range(nb_actions)])) for s in states}

plt.ion()                    
plot_blackjack(V)
plot_blackjack(greedy)
plt.ioff()
plt.show()  