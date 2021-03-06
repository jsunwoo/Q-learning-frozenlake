import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#import random as pr
#from gym.envs.registration import register

env = gym.make("FrozenLake-v0")

# Input and output size based on the Env
input_size = env.observation_space.n    # 16
output_size = env.action_space.n    # 4
learning_rate = 0.1

# These lines establish the feed-forward part of the network used to choose actions
X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)     # (1 * 16)
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))  # Initialize with random var (16 * 4)

Qpred = tf.matmul(X, W)     # Q-prediction (1 * 16) * (16 * 4) = (1 * 4)
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)    # (1 * 4)

loss = tf.reduce_sum(tf.square(Y - Qpred))  # cost(W) = (Ws - y) ^ 2
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)   # Minimize cost value

# Set other parameters
dis = 0.9
num_episodes = 2000
rList = []


def one_hot(x):
    return np.identity(16)[x:x + 1]


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        e = 1. / ((i / 50) + 10)
        rAll = 0
        done = False
        local_loss = []

        # The Q-Network training
        while not done:
            # Choose an action by greedily (with e chance of random action)
            Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})     # Same as asking to Q-table
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qs)

            # Get new state and reward from environment
            s1, reward, done, _ = env.step(action)

            if done:
                # Update Q, and no Qs+1, since it's a terminal state
                Qs[0, action] = reward      # row = 0, col = action
            else:
                # Obtain the Qs1 values by feeding the new state through our network
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})   # Same as asking to Q-table
                # Update Q
                Qs[0, action] = reward + dis * np.max(Qs1)

            # Train our network using target (Y) and predicted Q (Qpred) values
            sess.run(train, feed_dict={X: one_hot(s), Y: Qs})

            rAll += reward
            s = s1

        rList.append(rAll)

print("Success rate : " + str(sum(rList)/num_episodes))
print("Number of episodes : " + str(num_episodes))
