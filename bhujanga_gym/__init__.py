from gymnasium.envs.registration import register
from configparser import ConfigParser

config = ConfigParser()
config.read(r'bhujanga_gym\settings.ini')

board_width = int(config['GAME - BASIC']['BOARD_WIDTH'])
board_height = int(config['GAME - BASIC']['BOARD_HEIGHT'])

register(
    id="bhujanga_gym/SnakeWorld-v0",
    entry_point="bhujanga_gym.envs:SnakeWorldEnv",
    # max_episode_steps=board_width * board_height,
)
