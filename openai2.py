import gym
import matplotlib
import matplotlib.pyplot as plt
import time    
env = gym.make("ALE/Breakout-v5", render_mode='rgb_array')

from gym.utils import play

# play.play(env, zoom=3)    

#for steps in range(5):
#   env.render()

# env.close()

env.close()

array = env.render(mode='rgb_array')


plt.imshow(array)




