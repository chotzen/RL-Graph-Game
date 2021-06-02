import numpy as np

from gym import Env, spaces
from gym.utils import seeding

class Node:
    def __init__(self, ID, owner, units, neighbors):
        self.id = ID
        self.owner = owner
        self.units = units
        self.neighbors = neighbors

    def get_observation(self):
        return (self.owner, self.units, self.neighbors)


class Grid:
    def generate_edge_dict(N):
        for i in range(N):
            for j in range(N):
                ID = i*N + j
                

    def init_grid(N, init_amts):
        edge_dict = {}
        nodes = []
        for i in range(N):
            for j in range(N):
                ID = i * N + j
                edge_dict[ID] = set()
                if i != 0:
                    edge_dict[ID].add((i - 1) * N + j)
                if j != 0:
                    edge_dict[ID].add((i * N + j - 1)
                if i != N - 1:
                    edge_dict[ID].add((i + 1) * N + j)
                if j != N - 1: 
                    edge_dict[ID].add(i * N + j + 1)
                neighbors = [0] * N
                for x in edge_dict[ID]:
                    neighbors[x] = 1
                
                nodes.append(Node(2, init_amts[ID], tuple(neighbors)))
        
        nodes[0].owner = 0
        nodes[-1].owner = 1
        
        return nodes, edge_dict

class AI:
    def move(N):
        return spaces.MultiDiscrete([N, N]).sample()


class GraphGameEnv(Env):

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

    def __init__(self):
        self.N = 10
        self.MAX_U = 100
        self.players = 2
        self.neutral_l = 10
        self.neutral_h = 20
        self.start_units = 10
        self.action_space = spaces.MultiDiscrete([self.N, self.N])
        self.observation_space = spaces.Tuple(
            tuple([spaces.Tuple(
                (spaces.Discrete(self.players + 1), spaces.Discrete(self.MAX_U), spaces.MultiBinary(self.N)))
                ] * self.N))
        self.seed()
        self.nodes = None
        self.edges = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def perform_action(action, player):
        source = self.nodes[action[0]]
        target = self.nodes[action[1]]
        if source.owner == player and action[1] in self.edge_dict[action[0]]:
            if target.owner == player:
                target.units += source.units
            else:
                target.units -= source.units
                if target.units < 0:
                    target.owner = player
                    target.units *= -1
            

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        perform_action(action, 0)
        for i in range(1, players)
            perform_action(AI.move(self.N), i)
        
        counts = {}

        for x in self.nodes:
            if x.owner != players /
                x.units += 1
            
        
        obs = tuple([x.get_observation for x in self.nodes])




        


    
    def reset(self):
        node_vals = [self.np_random.randint(self.neutral_l, self.neutral_h + 1) for _ in range(self.N * self.N)]
        node_vals[0] = self.start_units
        node_vals[-1] = self.start_units
        self.nodes, self.edges = Grid.init_grid(self.N, node_vals)
        return tuple([x.get_observation for x in self.nodes])
        
    def render(self, mode='human'):
        for x in self.nodes:
            print(f"| {x.owner}, {x.units} |")
            if (x.id % self.N) = self.N - 1:
                print("\n")
    
    def close(self):
        pass


    



        
    
