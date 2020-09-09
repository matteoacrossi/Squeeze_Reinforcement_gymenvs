from gym.envs.registration import register

register(
    id='squeezeq-v0',
    entry_point='gym_squeezeq.envs:SqueezeEnvq',
)
