from .graph_util import *

import numpy as np
import gym

from gym import error, spaces, utils
from gym.utils import seeding

class GraphEnvR(gym.Env):

    """
    Description:
        A graph of nodes and edges, where each node holds a number of units.
    Source:
        see http://generals.io for inspiration
    Observation:
        Tuple (MultiDiscrete[3, 3, 3, ... (n times)], MultiDiscrete[MAX_TROOP, MT, MT, MT, ... (n times)], MultiBinary[n, n] (edges T/F))
    Actions:
        Type: MultiDiscrete[n, n]
    Reward:
        Reward is 1 for a winning termination, 0 otherwise (TO BE EDITED)
    Starting State:
        The edges are fixed and do not change (the multibinary[n, n] is fixed).
        Players start on opposite sides of a symmetric graph and 
    Episode Termination:
        All nodes in the MultiDiscrete[3, 3, 3, ...] are either 0 (player owned) or 1 (neutral) - reward 1
        All nodes in the MultiDiscrete[3, 3, 3, ...] are either 
    """

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, N, mode='random', model=None, reward_weights=(0, 0)):
        self.N = N # this N is the side of the board, not the total # of cells
        self.players = 2
        self.neutral_l = 10
        self.neutral_h = 20
        self.start_units = 10
        self.clock = 0
        self.growth_rate = 1
        self.mode = mode
        self.model = model
        self.reward_weights = reward_weights
        self.invalid_actions = 0
        # here, we reduce the dimensionality of the action space to 4n^2 < n^4
        # now coding inputs as (node, direction), with 0 -> up, 1 -> right, 2 -> down, 3 -> left
        # direction = a % 4
        # square = a // 4
        self.action_space = spaces.Discrete((self.N ** 2) * 4)
        self.observation_space = spaces.Tuple(
            tuple([spaces.Tuple(
                (spaces.Discrete(self.players + 1), spaces.Box(low=0, high=64000, shape=(1,), dtype=np.uint16), spaces.MultiBinary(self.N * self.N)))
                ] * (self.N * self.N)))
        self.seed()
        self.nodes = None
        self.edges = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action_to_nodes(self, action):
        source_idx = action // 4
        direction = action % 4

        # up
        if direction == 0 and source_idx - self.N >= 0:
            target_idx = source_idx - self.N

        # right
        elif direction == 1 and (source_idx + 1) % self.N != 0:
            target_idx = source_idx + 1

        # down
        elif direction == 2 and source_idx + self.N < self.N ** 2:
            target_idx = source_idx + self.N

        # left
        elif direction == 3 and (source_idx - 1) % self.N != self.N - 1:
            target_idx = source_idx - 1
        else:
            target_idx = source_idx


            
        source = self.nodes[source_idx]
        target = self.nodes[target_idx]
        return source_idx, target_idx, source, target


    def perform_action(self, action, player):
        # here, action should be a plain old number. I think
        source_idx, target_idx, source, target = self.action_to_nodes(action)
        if source.owner == player and target_idx in self.edges[source_idx]:
            if target.owner == player:
                target.units += source.units
            else:
                target.units -= source.units
                if target.units < 0:
                    target.owner = player
                    target.units *= -1
            source.units = 0
            
    def action_is_useful(self, action, player):
        '''
        Checks that the action given in the argument will actually do something. Useful for having 
        things actually happen if we're random sampling
        '''
        source_idx, target_idx, source, _ = self.action_to_nodes(action)
        return source.owner == player and target_idx in self.edges[source_idx]

    def flip_observation(self, obs, p1, p2):
        owner, units, nbs = obs 
        if owner == p2:
            return (p1, units, nbs)
        elif owner == p1:
            return (p2, units, nbs)
        return obs

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        reward = 0
        if not self.action_is_useful(action,0):
            reward -= 1000
            self.invalid_actions += 1
        
        # if self.mode == 'random':
        action2 = self.action_space.sample()
        while not self.action_is_useful(action2, 1):
            action2 = self.action_space.sample()

        # model.predict from a module

        self.perform_action(action, 0)
        self.perform_action(action2, 1)

        # TODO: add support for sampling from trained models, here. 
        

        players_left = set()
        for x in self.nodes:
            if x.owner != self.players: # not neutral
                if self.clock % self.growth_rate == 0:
                    x.units += 1 
                players_left.add(x.owner)
                if x.owner == 0:
                    reward += 50
                # reward += -1 * (x.owner - 0.5) * 2 * x.units * self.reward_weights[1]

        
        done = len(players_left) == 1 
        obs = tuple([x.get_observation() for x in self.nodes])
        # other_obs = tuple([self.flip_observation(x.get_observation(), 0,1) for x in self.nodes])
        if done:
            if 1 in players_left:
                reward += -10000
            else:
                reward += 10000

        self.clock += 1

        return obs, reward, done, {}

    def reset(self):
        node_vals = [self.np_random.randint(self.neutral_l, self.neutral_h + 1) for _ in range(self.N * self.N)]
        node_vals[0] = self.start_units
        node_vals[-1] = self.start_units
        self.nodes, self.edges = Grid.init_grid(self.N, node_vals)
        return tuple([x.get_observation() for x in self.nodes])
        
    def render(self, mode='human'):
        for x in self.nodes:
            print(f"| {x.owner}, {x.units} |", end="")
            if (x.id % self.N) == self.N - 1:
                print("\n")
        print("-------------------")
    
    def close(self):
        pass