import gym
import gym_graphs

env = gym.make('graphs-f-v0', N=5)
env.seed()
obs = env.reset()
env.render(mode='console')

done = False
total_reward = 0
rounds = 0

while not done:
    action1 = env.action_space.sample()
    while not env.action_is_useful(action1, 0):
        # print("hi")
        action1 = env.action_space.sample()
    

    obs1, reward, done, info = env.step(action1)
    total_reward += reward
    rounds += 1

    # env.render(mode='console')

print(total_reward)
print(rounds)