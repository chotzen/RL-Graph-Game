import numpy as np

class Node:
    def __init__(self, ID, owner, units, neighbors):
        self.id = ID
        self.owner = owner
        self.units = units
        self.neighbors = neighbors

    def get_observation(self):
        return (self.owner, np.array([self.units]), self.neighbors)


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
                    edge_dict[ID].add(i * N + j - 1)
                if i != N - 1:
                    edge_dict[ID].add((i + 1) * N + j)
                if j != (N - 1): 
                    edge_dict[ID].add(i * N + j + 1)
                
                neighbors = [0] * (N * N)
                for x in edge_dict[ID]:
                    neighbors[x] = 1
                
                nodes.append(Node(ID, 2, init_amts[ID], np.array(neighbors)))
        
        nodes[0].owner = 0
        nodes[-1].owner = 1
        
        return nodes, edge_dict