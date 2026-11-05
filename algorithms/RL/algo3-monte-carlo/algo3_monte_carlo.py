import numpy as np
import pandas as pd
from collections import defaultdict
import gymnasium as gym


num_timestep = 50000

env = gym.make("Blackjack-v1", natural=False, sab=False)


def policy(state):
    return 0 if state > 19 else 1


def generate_episode():
    episode = []
    state, _ = env.reset()
    # print(state)

    while True:
        action = policy(state[0])
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward)) 
        # print(next_state)

        if truncated or terminated: 
            # print("\n\nfinal next state: ", next_state)
            break

        state = next_state

    return episode


total_return = defaultdict(float)
N = defaultdict(int)
num_of_iterations = 1000

for i in range(num_of_iterations):
    episode = generate_episode()
    states, actions, rewards = zip(*episode)
    print(states)

    for t, state in enumerate(states):
        R = (sum(rewards[t:]))
        total_return[state] += R
        N[state] += 1

    print(total_return)
    # print(N)
    print("______-------______------_____-------_____")


# episode = generate_episode()
# print("\n", episode, "\n\n")






