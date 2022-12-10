import gym_super_mario_bros as sm

env = sm.make("SuperMarioBros-1-1-v0")
print(env.observation_space.shape)
print(env.action_space.n)