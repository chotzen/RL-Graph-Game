# -*- coding: utf-8 -*-
import os # turn off GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tqdm import tqdm

EPISODES = 1000
batch_size = 32

class GraphProcessor():
    def process_observation(observation):
        # print(len(observation))
        '''
        Converts the observation to an (N*N*3,) numpy array
        '''
        N = int(np.sqrt(len(observation)))
        assert N*N == len(observation)

        result = np.zeros((N, N, 3))
        for i, (owner, n_units, _) in enumerate(list(observation)):
            result[i // N, i % N, owner] = n_units

        return result.astype('uint16').flatten()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(60, input_dim=self.state_size, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('gym_graphs:graphs-v0')
    example_obs = GraphProcessor.process_observation(env.observation_space.sample())
    state_size = np.shape(example_obs)[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/graphs-dqn.h5")
    done = False
    for e in range(EPISODES):
        state = GraphProcessor.process_observation(env.reset())
        state = np.reshape(state, [1, state_size])
        tot_reward = 0
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(GraphProcessor.process_observation(next_state), [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            tot_reward += reward
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, tot_reward, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    agent.save("./save/cartpole-dqn-mod1.h5")