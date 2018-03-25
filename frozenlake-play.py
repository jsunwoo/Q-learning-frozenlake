import gym
import readchar

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT
}

# Set slippery mode
env = gym.make("FrozenLake-v0")
env.reset()

while True:
    env.render()
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print("Game aborted")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)

    if done:
        print("Finished with reward", reward)
        break
