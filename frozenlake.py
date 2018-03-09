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

# Discount factor (to find optimal(shortest) path)
dis = 0.99
# Set learning parameters
num_episodes = 100
rList = []

#a = np.arange(6).reshape(2,3)
#print(a)
#print(np.argmax(a))


for i in range(num_episodes):
    # Observe current state
    state = env.reset()
    rAll = 0
    done = False
    while not done:
        e= np.random.rand(1)
        if e < 0.2:
            action = rargmax(Q[state, :])
        else:
            action = env.action_space.sample()
        print(e, action)

        # action choose with decaying noise (to explore other way)
        #action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

        #print(Q)

        #if np.random.rand(1) < e:
        #e = np.random.rand(1)

        #action = rargmax(Q[state, :])
        #print(Q[state, :])
        #print(action)
        #print(np.argmax(Q[state, :]))
        #print(np.random.randn(1, env.action_space.n))
        #print(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

        # update new information
        new_state, reward, done, info = env.step(action)

        # Q(s,a) <- r + dis * maxQ(s', a')
        Q[state, action] = reward + dis * np.max(Q[new_state, :])
        rAll += reward

        # s <- s'
        state = new_state
        print "\nLeft(0) Down(1) Right(2) Up(3)",
        for o in range(16):
            if o % 4 == 0:
                print ''
            print "(",
            for a in range(4):
                print("%.2f" % Q[o][a]),
            print ")",
    rList.append(rAll)


print("\nSuccess rate : " + str(sum(rList)/num_episodes))
