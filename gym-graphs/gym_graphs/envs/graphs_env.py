from .graph_util import *

import numpy as np
import gym

from gym import error, spaces, utils
from gym.utils import seeding


class AI:
    def move(N):
        return spaces.MultiDiscrete([N*N, N*N]).sample()


class GraphsEnv(gym.Env):

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

    def __init__(self, N, mode='random', model=None):
        self.N = N
        self.players = 2
        self.neutral_l = 10
        self.neutral_h = 20
        self.start_units = 10
        self.clock = 0
        self.growth_rate = 5
        self.mode = mode
        self.model = model
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
            
    def action_is_useful(self, action, player):
        '''
        Checks that the action given in the argument will actually do something. Useful for having 
        things actually happen if we're random sampling
        '''
        source = self.nodes[action[0]]
        target = self.nodes[action[1]]
        return source.owner == player and action[1] in self.edges[action[0]]

    def flip_observation(self, obs, p1, p2):
        owner, units, nbs = obs 
        if owner == p2:
            return (p1, units, nbs)
        elif owner == p1:
            return (p2, units, nbs)
        return obs

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        print(action)
        assert self.action_space.contains(action), err_msg

        self.perform_action(action, 0)
        # if self.mode == 'random':
        action2 = self.action_space.sample()
        while not self.action_is_useful(action2, 1):
            action2 = self.action_space.sample()

        # TODO: add support for sampling from trained models, here. 
        
        players_left = set()
        for x in self.nodes:
            if x.owner != self.players: # not neutral
                if self.clock % self.growth_rate == 0:
                    x.units += 1 
                players_left.add(x.owner)
        
        done = len(players_left) == 1    
        obs = tuple([x.get_observation() for x in self.nodes])
        # other_obs = tuple([self.flip_observation(x.get_observation(), 0,1) for x in self.nodes])
        if done:
            if 1 in players_left:
                reward = -1
            else:
                reward = 1
        else:
            reward = 0

        self.clock += 1

        return obs, reward, done, {}

    def reset(self):
        node_vals = [self.np_random.randint(self.neutral_l, self.neutral_h + 1) for _ in range(self.N * self.N)]
        node_vals[0] = self.start_units
        node_vals[-1] = self.start_units
        self.nodes, self.edges = Grid.init_grid(self.N, node_vals)
        return tuple([x.get_observation() for x in self.nodes])
        
    def render(self, mode='human'):
        # viewer = SimpleImageViewer()
        viewer = rendering.Viewer(500,500)
        for x in self.nodes:
            print(f"| {x.owner}, {x.units} |", end="")
            if (x.id % self.N) == self.N - 1:
                print("\n")
        if mode == 'graphic':
            heatMap = np.zeros()
            for x in self.nodes:
                heatmap[x.get_observation()[0]][x.get_observation()[1]][0] = x.units
            colors = [[Color(heatmap[i,j,:] for j in range(self.N))] for i in range(self.N)]
            viewer.imshow(colors)
            # viewer.close()

        
        elif mode == 'console':
            for x in self.nodes:
                print(f"| {x.owner}, {x.units} |", end="")
                if (x.id % self.N) == self.N - 1:
                    print("\n")
            
            heatMap = np.zeros()
            for x in self.nodes:
                heatmap[x.get_observation()[0]][x.get_observation()[1]][0] = x.units
            colors = [[Color(heatmap[i,j,:] for j in range(self.N))] for i in range(self.N)]
            viewer.imshow(colors)
            # viewer.close()
            return viewer.render(return_rgb_array = mode=='rgb_array')
    
    def close(self):
        pass