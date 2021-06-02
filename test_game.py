# from graph_game import GraphGameEnv
# import numpy as np

# def main():
#     env = GraphGameEnv()
#     env.seed()
#     init_obs = env.reset()
#     env.render()

#     while True:
#         source = int(input("Source: "))
#         target = int(input("Target: "))
#         obs, reward, done, aux = env.step(np.array([source, target]))
#         env.render()
#         if done:
#             break

# main()
