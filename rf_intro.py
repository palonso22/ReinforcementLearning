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

EPOCHS = 20000
ALPHA = 0.8
GAMMA = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.0
decay_rate = 0.001
log_interval = 1000

def epsilon_greedy_action_selection(epsilon, q_table, discrete_state):
    random_number = np.random.random()

    # explotaition

    if random_number > epsilon:
        state_row = q_table[discrete_state,:]
        return np.argmax(state_row)

    # exploration    
    else:
        return env.action_space.sample()


def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
     return old_q_value + ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)


def reduce_epsilon(epsilon, epoch):
    return min_epsilon + (max_epsilon-min_epsilon)*np.exp(- decay_rate * epoch)


rewards = []

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
fig.canvas.draw()
epoch_plot_tracker = []
total_reward_plot_tracker = []


for episode in range(EPOCHS):

    state = env.reset()
    done = False
    total_rewards = 0

    while not done:

        action = epsilon_greedy_action_selection(epsilon, q_table, state)

        new_state, reward, done, info = env.step(action)

        old_q_value = q_table[state, action]

        next_optimal_q_value = np.max(q_table[new_state, :])

        next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)

        q_table[state, action] = next_q

        total_rewards += reward

        state = new_state

        
    
    epsilon = reduce_epsilon(epsilon, episode)

    rewards.append(total_rewards)

    total_reward_plot_tracker.append(np.sum(rewards))
    epoch_plot_tracker.append(episode)


    if episode % log_interval == 0:
        ax.clear()
        ax.plot(epoch_plot_tracker, total_reward_plot_tracker)
        fig.canvas.draw()


# state = env.reset()
# for step in range(100):    
#     env.render()
#     action = np.argmax(q_table[state, :])
#     state, reward, done, info = env.step(action)
#     time.sleep(1)
#     os.system('clear')
#     if done:
#         break

env.close()