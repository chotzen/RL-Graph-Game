from .graph_util import *

import numpy as np
import gym

from gym import error, spaces, utils
from gym.utils import seeding


class AI:
    def move(N):
        return spaces.Discrete(N*N*4).sample()


class GraphsEnv(gym.Env):

    """
    Description:
        A graph of nodes and edges, where each node holds a number of units.
    Source:
        see http://generals.io for inspiration
    Observation:
        Tuple (MultiDiscrete[3, 3, 3, ... (n times)], MultiDiscrete[MAX_TROOP, MT, MT, MT, ... (n times)], MultiBinary[n, n] (edges T/F))
        First describes owner: 1 = agent, 2 = ai, 3 = neutral.
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

    def __init__(self):
        self.N = 3
        self.max_degree = 4
        self.players = 2
        self.neutral_l = 10
        self.neutral_h = 20
        self.start_units = 10
        self.action_space = spaces.Discrete(self.N * self.N * self.max_degree)
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

    def perform_action(self, action, player): # returns boolean on valididty of action
        sourceid, targetid = Grid.get_action_IDs(action, self.N)

        if targetid < 0 or targetid >= len(self.nodes):
            return False

        source = self.nodes[sourceid]
        target = self.nodes[targetid]

        if source.owner == player and target.id in self.edges[source.id]:
            if target.owner == player:
                target.units += source.units
            else:
                target.units -= source.units
                if target.units < 0:
                    target.owner = player
                    target.units *= -1
            source.units = 0
            return True

        return False
            

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        valid_act = self.perform_action(action, 0)
        
        self.perform_action(AI.move(self.N), 1)
        
        players_left = set()
        for x in self.nodes:
            if x.owner != self.players: # not neutral
                x.units += 1
                players_left.add(x.owner)
        
        done = len(players_left) == 1    
        obs = tuple([x.get_observation() for x in self.nodes])
        if done and 0 in players_left:
            reward = 100
        elif done and 1 in players_left:
            reward = -50
        elif not valid_act:
            reward = -1 # penalize invalid play
        else:
            reward = 0 # survival is nice

        return obs, reward, done, 0

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
        print("--------------------------")
    
    def close(self):
        pass