import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make("FrozenLake-v3")

# initial board
env.render()

# initialize table with all zeros
# space : 16, action : 4
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
num_episodes = 100

rList = []
for i in range(num_episodes):
    # Observe current state
    state = env.reset()
    rAll = 0
    done = False
    while not done:
        # if Q values are same then pick the way randomly
        action = rargmax(Q[state, :])
        # update new information
        new_state, reward, done, info = env.step(action)
        # Q(s,a) <- r + maxQ(s', a')
        Q[state, action] = reward + np.max(Q[new_state, :])
        rAll += reward
        # s <- s'
        state = new_state
    rList.append(rAll)
    print(" Left Down Right Up")
    print(Q)

print("Success rate : " + str(sum(rList)/num_episodes))