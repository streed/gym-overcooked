import gym
import gym_overcooked

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.optimizers import Adam

import gym
import numpy as np
import random

from collections import deque

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import backend as K

class DeepQLearning:
    def __init__(self, state_size, action_size, memory_length=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_length)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.75
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(4,
                         (16, 16),
                         strides=16,
                         input_shape=self.state_size,
                         activation='relu'))
        model.add(Conv2D(8, (4, 4), strides=4, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        if state is None:
            return
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if state is None:
            return random.randrange(self.action_size)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = gym.make('overcooked-v0')

state_size = env.observation_space.shape[0]

agent = DeepQLearning(env.observation_space.shape, env.action_space.n)

episodes = 100
steps = 100

batch_size = 5

for e in range(1, episodes):
    state = env.reset()
    for time in range(steps):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.expand_dims(np.copy(next_state), axis=0)
        print("Episode: {} Time: {} Reward: {} Epsilon: {}".format(e, time, reward, agent.epsilon))
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

agent.model.save_weights('overcooked.h5')
