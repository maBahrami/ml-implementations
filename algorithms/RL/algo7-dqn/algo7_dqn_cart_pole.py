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
            action_values = self.model.predict(state)[0]    # returns batch of values of each action at a batch of states
            return np.argmax(action_values)

    def train(self):
        pass


nS = env.observation_space.shape[0]
nA = int(env.action_space.n)

dqn = DQN(nS, nA, alpha=0.001, gamma=0.95, epsilon=1, epsilon_min=0.001, epsilon_decay=0.995)

