from statistics import mode
import gym
import matplotlib
import matplotlib.pyplot as plt
import time    

env_name = 'MountainCar-v0'
env = gym.make(env_name)

from gym.utils import play

env.seed(42)
observation = env.reset()

def simple_agent(observation) -> int:
    position, velocity = observation

    # When go to the right
    if -0.1 < position < 0.4 :
        print('a')
        return 2        

    # When go to the left

    elif velocity < 0 and position < -0.2: 
        print('b')
        return 0

     
    # When to not do anything
    else:
        print('c')
        return 1         

for step in range(300):
    env.render()
    action = simple_agent(observation)
    observation, reward, done, info = env.step(action)    
    print(f"position {observation[0]}")
    print(f"velocity {observation[1]}")
    print(f"reward {reward}")
    print(f"info {info}")
    time.sleep(0.01)



