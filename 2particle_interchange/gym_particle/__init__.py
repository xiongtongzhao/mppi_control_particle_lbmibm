from gym.envs.registration import register

register(
    id='particle-v0',
    entry_point='gym_particle.envs:ParticleEnv',
    reward_threshold=1.0,
)

register(
    id='traj-v0',
    entry_point='gym_particle.envs:ParticleEnv_t',
    reward_threshold=1.0,
)