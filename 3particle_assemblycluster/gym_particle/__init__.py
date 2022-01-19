from gym.envs.registration import register

register(
    id='particle-v0',
    entry_point='gym_particle.envs:ParticleEnv',
    reward_threshold=1.0,
)

