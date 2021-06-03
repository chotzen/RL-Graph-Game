import gym
import gym_graphs

env = gym.make('graphs-2p-v0', N=5)
env.seed()
obs = env.reset()
env.render(mode='console')

done = False
total_reward = 0
rounds = 0

while not done:
    action1 = env.action_space.sample()
    while not env.action_is_useful(action1, 0):
        action1 = env.action_space.sample()

    action2 = env.action_space.sample()
    while not env.action_is_useful(action2, 1):
        action2 = env.action_space.sample()


    obs1, reward, done, info = env.step(action1, action2)
    total_reward += reward
    rounds += 1

    env.render(mode='console')

print(total_reward)
print(rounds)