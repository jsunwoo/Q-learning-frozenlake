import gym
import numpy as np
from gym.envs.registration import register
import random as pr

# Set slippery mode
env = gym.make("FrozenLake-v0")

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
dis = 0.9
num_episodes = 2000
learning_rate = 0.85

rList = []
for i in range(num_episodes):
    # Observe current state
    state = env.reset()
    rAll = 0
    done = False

    # The Q-Table learning algorithm
    while not done:
        # Choose an action by greedily (with noise)
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

        # Update new information
        new_state, reward, done, info = env.step(action)

        # Q(s,a) <- (1-a) * Q(s,a) + a * (r + dis * maxQ(s', a'))
        # Keep 90% of my knowledge and use 10% of Q's direction
        Q[state, action] = (1 - learning_rate) * Q[state, action]\
                           + learning_rate * (reward + dis * np.max(Q[new_state, :]))

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
