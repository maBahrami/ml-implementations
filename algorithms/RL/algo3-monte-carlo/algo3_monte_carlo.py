import numpy as np
import pandas as pd
from collections import defaultdict
import gymnasium as gym


num_of_episodes = 50000

env = gym.make("Blackjack-v1", natural=False, sab=False)


def policy_1(state):                  # considering any initial policy out of nowhere
    return 0 if state > 19 else 1


def generate_episode(policy):            # geneating an episode according ot the policy
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


def predict_return(policy, first_visit=True):       # first_visit = True  --> first-visit
                                            # first_visit = False --> every-visit
    total_return = defaultdict(float)
    N = defaultdict(int)

    for i in range(num_of_episodes):
        episode = generate_episode(policy)        # episode is a list like: [(state, action, reward), . . . ]
                                             # and state is like: (sum of our cards, dealer's card, ace)
        states, actions, rewards = zip(*episode)

        for t, state in enumerate(states):
            if first_visit and state in states[:t]:
                continue
            else:
                R = sum(rewards[t:])
                total_return[state] += R
                N[state] += 1

        if i % 10000 == 0:
            print(f"[INFO]   {i} iterations passed")


    return total_return, N



total_return, N = predict_return(policy=policy_1, first_visit=True)
