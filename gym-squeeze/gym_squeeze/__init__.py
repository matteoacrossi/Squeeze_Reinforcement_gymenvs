from gym.envs.registration import register

register(
    id='squeeze-v0',
    entry_point='gym_squeeze.envs:SqueezeEnv',
)
