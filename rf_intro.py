import numpy as np
import matplotlib.pyplot as plt
import gym 
import time
import os

from gym.envs.registration import register

env_name = 'FrozenLakeNotSlippery-v0'
try:
    register(
    id = env_name,
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
    )
except:
    print('Already registered')    


env = gym.make(env_name)    
env.reset()


action_size =  env.action_space.n
state_size = env.observation_space.n

q_table = np.zeros([state_size, action_size])

print(q_table)

EPOCHS = 20000
ALPHA = 0.8
GAMMA = 0.95

for step in range(500):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)    
    time.sleep(0.3)
    os.system('clear')
    if done:
        env.reset()

env.close()    