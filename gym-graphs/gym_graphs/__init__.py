from gym.envs.registration import register

register(
    id='graphs-v0',
    entry_point='gym_graphs.envs:GraphsEnv',
)

register(
    id='graphs-2p-v0',
    entry_point='gym_graphs.envs:GraphEnv2P',
)

register(
    id='graphs-f-v0',
    entry_point='gym_graphs.envs:GraphEnvF',
)