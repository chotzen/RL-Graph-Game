from .graph_util import *

import numpy as np

from gym import Env, spaces
from gym.utils import seeding

from gym.envs.classic_control import rendering

class GraphEnv2P(Env):

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
        'render.modes': ['console, graphic']
    }

    def __init__(self, N):
        self.N = N
        self.players = 2
        self.neutral_l = 10
        self.neutral_h = 20
        self.start_units = 10
        self.action_space = spaces.MultiDiscrete([self.N * self.N, self.N * self.N])
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

    def action_is_useful(self, action, player):
        '''
        Checks that the action given in the argument will actually do something. Useful for having 
        things actually happen if we're random sampling
        '''
        source = self.nodes[action[0]]
        target = self.nodes[action[1]]
        return source.owner == player and action[1] in self.edges[action[0]]


    def perform_action(self, action, player):
        source = self.nodes[action[0]]
        target = self.nodes[action[1]]
        if source.owner == player and action[1] in self.edges[action[0]]:
            if target.owner == player:
                target.units += source.units
            else:
                target.units -= source.units
                if target.units < 0:
                    target.owner = player
                    target.units *= -1
            source.units = 0

    def flip_observation(self, obs, p1, p2):
        owner, units, nbs = obs

            

    def step(self, action1, action2):
        '''
        This step function takes two actions, one from each player, and applies both of them. 
        Since the troop counts in a node are just the sum of the movements into it, there is no preference to which player moved first. 
        '''
        err_msg = "%r (%s) invalid" % (action1, type(action1))
        err_msg = "%r (%s) invalid" % (action2, type(action2))
        assert self.action_space.contains(action1), err_msg
        assert self.action_space.contains(action2), err_msg

        self.perform_action(action1, 0)
        self.perform_action(action2, 1)
        
        players_left = set()
        for x in self.nodes:
            if x.owner != self.players: # not neutral
                x.units += 1
                players_left.add(x.owner)

        done = len(players_left) == 1    
        obs = tuple([x.get_observation() for x in self.nodes])
        other_obs = tuple([Node.flip_observation(x.get_observation()) for x in self.nodes])
        reward = 1.0 if done and 0 in players_left else 0.0

        return obs, reward, done, {'other_obs': other_obs}

    def reset(self):
        node_vals = [self.np_random.randint(self.neutral_l, self.neutral_h + 1) for _ in range(self.N * self.N)]
        node_vals[0] = self.start_units
        node_vals[-1] = self.start_units
        self.nodes, self.edges = Grid.init_grid(self.N, node_vals)
        print(self.edges)
        return tuple([x.get_observation() for x in self.nodes])
        
    def render(self, mode='console'):
        #here
        if mode == 'graphic':
            heatMap = np.zeros()
            for x in self.nodes:
                heatmap[x.observation_space[0]][x.observation_space[1]][0] = x.units
                
        
        elif mode == 'console':
            for x in self.nodes:
                print(f"| {x.owner}, {x.units} |", end="")
                if (x.id % self.N) == self.N - 1:
                    print("\n")
    
    def close(self):
        pass