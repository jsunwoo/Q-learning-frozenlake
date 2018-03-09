import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

# random arguments of the maxima
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

# Discount factor (to find optimal(shorter) path)
dis = 0.9

# Set learning parameters
num_episodes = 1000
rList = []

for i in range(num_episodes):
    # Observe current state
    state = env.reset()
    rAll = 0
    done = False
    while not done:
        # E-greedy (to explore)
        if np.random.rand(1) > 1.0 / (i + 1):
            action = rargmax(Q[state, :])
        else:
            action = env.action_space.sample()

        # update new information
        new_state, reward, done, info = env.step(action)

        # Q(s,a) <- r + dis * maxQ(s', a')
        Q[state, action] = reward + dis * np.max(Q[new_state, :])
        rAll += reward

        # s <- s'
        state = new_state
    rList.append(rAll)

print "Left(0) Down(1) Right(2) Up(3)\n",
for o in range(16):
    if o % 4 == 0:
        print ''
    print "(",
    for a in range(4):
        print("%.2f" % Q[o][a]),
    print ")",
print("\n\nSuccess rate : " + str(sum(rList)/num_episodes))
print("Number of episodes : " + str(num_episodes))
