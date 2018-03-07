import gym
from gym.envs.registration import register
import sys, tty, termios

# 키 받아오기 getChar
class _GetCh:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
inkey = _GetCh()

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
    '\x1b[D': LEFT,
}

# 설정하는 부분
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

# environment를 만들어주기
env = gym.make("FrozenLake-v3")
# initial board
# render는 말그대로 environment를 그려주는 역할
env.render()

while True:
    # 키를 받아주고
    key = inkey()

    # 키가 만약 상하좌우가 아니면 게임 종료
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    # action에 방향키를 받아온다
    action = arrow_keys[key]
    # action을 취함으로써 정보들을 갱신해준다
    state, reward, done, info = env.step(action)
    env.render()
    # 헷갈리지 않게 정보를 계속 출력해주기 위함
    print("State: ", state, "Action: ", action,
          "Reward: ", reward, "Info: ", info)

    # 게임이 종료되었을 경우에 해당
    if done:
        print("Finished with reward", reward)
        break
