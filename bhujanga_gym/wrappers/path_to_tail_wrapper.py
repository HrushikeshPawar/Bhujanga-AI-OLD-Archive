
# Imort wrapper environment
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from bhujanga_gym.envs.snake_world import SnakeWorldEnv
from bhujanga_ai.helper import BFS_Finder
import numpy as np
from numpy._typing import NDArray
from typing import Optional
from collections import deque


# Path to tail Wrapper
class PathToTailWrapper(ObservationWrapper):

    # Constructor
    def __init__(self, env: SnakeWorldEnv):

        # Call super constructor
        super().__init__(env)
        self.env = env

        # Set the observation space
        self.observation_space = Box(low=0, high=2, shape=(21,), dtype='uint8')

        # Setup path finder
        self.path_finder = BFS_Finder(env)

    # Find if path to the tail exists
    def find_path_to_tail(self) -> int:

        # Get the snake's head position
        head_pos = self.env.snake.head

        # Get the snake's tail position
        tail_pos = self.env.snake.tail

        # Find the path
        self.path_finder.find_path(head_pos, tail_pos)

        # Return if path exists
        return int(self.path_finder.path_exists())

    # Observation
    def observation(self, observation) -> NDArray:

        path_to_tail = None

        if self.env.snake.score < min(self.env.board_height, self.env.board_width):
            path_to_tail = 1
        else:
            path_to_tail = self.find_path_to_tail()

        # Append this to the observation
        observation = np.append(observation, path_to_tail)

        return observation


# Path to tail Stacked Wrapper
class PathToTailStackedWrapper(ObservationWrapper):

    # Constructor
    def __init__(self, env: SnakeWorldEnv, stack_size: Optional[int] = 2):

        # Call super constructor
        super().__init__(env)
        self.env = env
        self.stack_size = stack_size

        # Set the observation space
        self.observation_space = Box(low=0, high=2, shape=(stack_size, 21), dtype='uint8')

        # Previous view
        self.previous_view = deque(maxlen=stack_size)

        # Setup path finder
        self.path_finder = BFS_Finder(env)

    # Find if path to the tail exists
    def find_path_to_tail(self) -> int:

        # Get the snake's head position
        head_pos = self.env.snake.head

        # Get the snake's tail position
        tail_pos = self.env.snake.tail

        # Find the path
        self.path_finder.find_path(head_pos, tail_pos)

        # Return if path exists
        return int(self.path_finder.path_exists())

    # Observation
    def observation(self, observation) -> NDArray:

        path_to_tail = None

        if self.env.snake.score < min(self.env.board_height, self.env.board_width):
            path_to_tail = 1
        else:
            path_to_tail = self.find_path_to_tail()

        # Append this to the observation
        observation = np.append(observation, path_to_tail)

        # if previous observations are less then stack the current observation over and over
        while len(self.previous_view) < self.stack_size - 1:
            self.previous_view.append(observation)

        # Append current view to previous view
        self.previous_view.append(observation)

        # Stack the views
        return np.stack((self.previous_view))
