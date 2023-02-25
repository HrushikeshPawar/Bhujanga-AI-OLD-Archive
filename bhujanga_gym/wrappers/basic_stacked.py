# Imort wrapper environment
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from bhujanga_gym.envs.snake_world import SnakeWorldEnv
import numpy as np
from numpy._typing import NDArray
from typing import Optional
from collections import deque


# Basic Stacked Wrapper
class BasicStackedWrapper(ObservationWrapper):

    # Constructor
    def __init__(self, env: SnakeWorldEnv, stack_size: Optional[int] = 2):

        # Call super constructor
        super().__init__(env)
        self.env = env
        self.stack_size = stack_size

        # Set the observation space
        self.observation_space = Box(low=0, high=2, shape=(stack_size, 20), dtype='uint8')

        # Previous view
        self.previous_view = deque(maxlen=stack_size)

    # Observation
    def observation(self, observation) -> NDArray:

        # if previous observations are less then stack the current observation over and over
        while len(self.previous_view) < self.stack_size - 1:
            self.previous_view.append(observation)

        # Append current view to previous view
        self.previous_view.append(observation)

        # Stack the views
        return np.stack((self.previous_view))
