import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
env.close()


# graph, collection of nodes and edges
# Node: keeps track of who controls (or neutral), number of units present
# Edge: should be weighted with a discrete weight, represents the amount of ticks needed to traverse it.
# for now, just have instantaneous travel

# action space of player: at any step, the player can specify a node under their control and a target node, 
# sending all units present along the weighted edge to that target

# observations: players can know the whole map, we'll start with this
# Another possibility is perhaps they can only know 'one layer' of edges/nodes away from nodes they own, or maybe even just the edge.

# at every step, every player/opponent controlled node gains 1 unit. 

# step takes an action. An action is simply two nodes a source and a target, the source must be under the player's control
# step calls a black box function representing the AI, which also gives the move of the opponent.
# stng state, neutral nodes should have a arbitrary 'cost' to take over, say 25. 

# playing itself

# playing multiple versions of itself? <- fighting someone is bad for both of you

#lets just start with a grid as proof of concept.

# reward??
# 1 always? 0 unless win 1?

# action space: discrete, any node the player owns, combined with a neighboring node.
# multi-discrete - 2 n-discrete, where n is number of nodes
# illegal moves do not do anything.
# observation
# Tuple of n tuples
# each of these n tuples has the following
# discrete 3, where 0 is player owned, 1 is enemy owned, 2 is neutral.
# discrete MAX_TROOP, representing the number of troops at the node.
# multi-binary(n), representing edges 0 means no edge, 1 means edge.


# store graph as dictionary mapping node to a list of other 