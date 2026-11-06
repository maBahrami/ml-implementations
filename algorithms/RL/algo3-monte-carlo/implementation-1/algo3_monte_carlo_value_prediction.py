import numpy as np
import pandas as pd
from collections import defaultdict
import gymnasium as gym


env = gym.make("Blackjack-v1", natural=False, sab=False)


def policy_1(state):
    """considering any initial policy out of nowhere"""
    return 0 if state > 19 else 1


def generate_episode(policy):
    """geneating an episode according ot the policy. """
    episode = []
    state, _ = env.reset()
    # print(state)

    while True:
        action = policy(state[0])
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward)) 
        # print(next_state)

        if truncated or terminated: break

        state = next_state

    return episode


def predict_value(policy, method, num_of_episodes):
    """Estimate state returns with Monte Carlo.

    method = "first_visit"  -> first-visit MC
    method = "every_visit"  -> every-visit MC
    """

    total_return = defaultdict(float)
    N = defaultdict(int)

    if method not in ("first_visit", "every_visit"):
        raise ValueError(f"unknown method: {method}")

    for i in range(num_of_episodes):
        # episode is a list like: [(state, action, reward), . . . ]
        # and state is like: (sum of our cards, dealer's card, ace)
        episode = generate_episode(policy)                          
        states, _, rewards = zip(*episode)

        for t, state in enumerate(states):
            if method == "first_visit" and state in states[:t]:
                continue

            R = sum(rewards[t:])
            total_return[state] += R
            N[state] += 1

        if i % 10000 == 0:
            print(f"[INFO]   {i} iterations passed")

    total_return = pd.DataFrame(total_return.items(), columns=["state", "total_return"])
    N = pd.DataFrame(N.items(), columns=["state", "N"])
    df = pd.merge(total_return, N, on="state")
    df["value"] = df["total_return"] / df["N"]

    return df


df = predict_value(policy=policy_1, method="every_visit", num_of_episodes=500000)

df = df.sort_values("value", ascending=False)
print(df.head(10))
print(df.tail(10))
# print(df[df["state"] == (18, 5, 0)]["value"])