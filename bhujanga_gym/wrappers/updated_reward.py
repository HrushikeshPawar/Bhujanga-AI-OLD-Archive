# Imort wrapper environment
from gymnasium import Wrapper
from bhujanga_gym.envs.snake_world import SnakeWorldEnv
from bhujanga_ai.helper import BFS_Finder
from numpy._typing import NDArray


# Updated Reward Wrapper
class UpdatedRewardWrapper(Wrapper):

    # Constructor
    def __init__(self, env: SnakeWorldEnv):
        super().__init__(env)
        self.env = env

        # Setup path finder
        self.path_finder = BFS_Finder(env)

    # Find if path to the tail exists
    def find_path_to_tail(self) -> bool:

        # Get the snake's head position
        head_pos = self.env.snake.head

        # Get the snake's tail position
        tail_pos = self.env.snake.tail

        # Find the path
        self.path_finder.find_path(head_pos, tail_pos)

        # Return if path exists
        return self.path_finder.path_exists()

    # Step
    def step(self, action: int) -> tuple[NDArray, float, bool, bool, dict]:

        # Get the observation, reward, done, truncated and info
        obs, reward, done, trun, info = self.env.step(action)

        if info['collision']:
            reward = -3

        elif info['truncated']:
            reward = -5

        elif self.env.snake.score >= min(self.env.board_height, self.env.board_width):
            if self.find_path_to_tail():
                reward += 0.5

            elif not self.find_path_to_tail() and self.env.snake.score >= min(self.env.board_height, self.env.board_width):
                reward -= 2

        # return the updated reward
        return obs, reward, done, trun, info
