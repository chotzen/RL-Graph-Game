import os # turn off GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
from gym.core import ObservationWrapper
import numpy as np

import gym
import gym_graphs

# import wandb
# from wandb.keras import WandbCallback

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras import Input
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class GraphProcessor(Processor):
    def process_observation(self, observation):
        # print(len(observation))
        '''
        Converts the observation to an (N, N, 3) numpy array
        '''
        N = int(np.sqrt(len(observation)))
        assert N*N == len(observation)

        result = np.zeros((N, N, 3))
        for i, (owner, n_units, _) in enumerate(list(observation)):
            result[i // N, i % N, owner] = n_units

        return result.astype('uint8')

    def process_state_batch(self, batch):
        return np.squeeze(batch, axis=1)

# wandb.init(project="graph-rl")
        
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=5)
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--env-name', type=str, default='1')
args = parser.parse_args()

env = gym.make('graphs-r-v0', N=args.n, mode='random', reward_weights=(1000, 0)) # TODO: add more parameters here so we can grid search
env.seed()
obs = env.reset()
env.render(mode='console')

C1 = 20
C2 = 40
C3 = 40
D = 200

'''
outline of how this is going to work:
- first, train a model against a random agent. hopefully, it will beat it
- call the trained model model_0
- then, train a model against model_0, and fix it. train a model against that one.
- continue this until they're too good (how do we evaluate this?)
'''

model = Sequential()
model.add(Dense(D, input_shape=(args.n, args.n, 3)))
# model.add(Convolution2D(C1, (2, 2), input_shape=(args.n, args.n, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(D))
model.add(Activation('relu'))
model.add(Dense(D))
model.add(Activation('relu'))
model.add(Dense(env.action_space.n))
model.add(Activation('linear'))
model.build()
print(model.summary())

processor = GraphProcessor()

policy = EpsGreedyQPolicy() # tinker with this

dqn = DQNAgent(model=model, nb_actions=env.action_space.n, policy=policy, memory=EpisodeParameterMemory(100000, window_length=1),
            processor=processor, nb_steps_warmup=150000, gamma=.95, target_model_update=10000,
            train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=.00025), metrics=['mae']) # also tinker with learning rate here

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=100000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)

elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)

