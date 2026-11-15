import random
import gymnasium as gym
import numpy as np
from collections import deque
from keras import models, layers
from keras.optimizers import Adam

env = gym.make("CartPole-v1")

M = 500     # episode
T = 500     # step
batch_size = 24

class DQN():
    def __init__(self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.gamma = gamma
        self.memory = deque([], maxlen=2500)
        self.nS = states
        self.nA = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.loss = []

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.nS,)),
            layers.Dense(24, activation="relu"),
            layers.Dense(24, activation="relu"),
            layers.Dense(self.nA, activation="linear")
        ])

        model.compile(optimizer=Adam(learning_rate=self.alpha), loss="mse")
        return model

    def epsilon_greedy(self, state):
        if random.random() <= self.epsilon:
            return env.action_space.sample()
        else:
            # predict returns Q-values for a batch, shape (1, nA); [0] takes the single row
            action_values = self.model.predict(state)[0]
            return np.argmax(action_values)

    def store(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def train(self):
        
        pass


nS = env.observation_space.shape[0]
nA = int(env.action_space.n)

dqn = DQN(nS, nA, alpha=0.001, gamma=0.95, epsilon=1, epsilon_min=0.001, epsilon_decay=0.995)

for episode in range(M):
    state, _ = env.reset()
    state = np.reshape(state, (1, dqn.nS))  # input of the NN has to be 2D
    total_reward = 0    # to monitor during training

    for t in range(T):
        action = dqn.epsilon_greedy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.reshape(next_state, (1, dqn.nS))
        total_reward += reward

        dqn.store(state, action, reward, next_state, terminated)

        if len(dqn.memory) > batch_size:
            dqn.train()

        state = next_state

        if terminated or truncated:
            print(f"episode: {episode}, score: {total_reward}, epsilon: {dqn.epsilon}")
            break

