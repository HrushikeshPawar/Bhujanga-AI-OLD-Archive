
# Imort wrapper environment
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from bhujanga_gym.envs.snake_world import SnakeWorldEnv
import numpy as np
from numpy._typing import NDArray
from typing import Optional
from collections import deque


# Full View Wrapper
class FullViewWrapper(ObservationWrapper):

    # Constructor
    def __init__(self, env: SnakeWorldEnv):

        # Call super constructor
        super().__init__(env)
        self.env = env

        # Get the board height and width
        board_height = env.board_height
        board_width = env.board_width

        # Set the observation space
        self.observation_space = Box(low=0, high=5 * board_width, shape=(2, board_height, board_width), dtype='uint8')

    # Observation
    def observation(self, observation) -> NDArray:

        # Get Food Position
        food_pos = self.env.food.position

        # Get Snake's Head Position
        snake_head_pos = self.env.snake.head

        # Get Snake's Body Position
        snake_body_pos = self.env.snake.body

        # Create any np array of the same shape as the observation
        # and fill it with zeros
        full_view = np.zeros(self.observation_space.shape, dtype='uint8')

        # Set the food position to 1
        full_view[0, food_pos.y, food_pos.x] = 1

        # Set the snake's head position to 1
        try:
            full_view[1, snake_head_pos.y, snake_head_pos.x] = 5
        except IndexError:
            # Means snake has crossed the boundary
            pass

        # Set the snake's body position to 1
        for pos in snake_body_pos:
            full_view[1, pos.y, pos.x] = 1

        return full_view


# Full View with Stacking Wrapper
class FullViewWithStackWrapper(ObservationWrapper):

    # Constructor
    def __init__(self, env: SnakeWorldEnv, stack_size: Optional[int] = 2):

        # Call super constructor
        super().__init__(env)
        self.env = env

        # Get the board height and width
        board_height = env.board_height
        board_width = env.board_width

        # Set the observation space
        self.observation_space = Box(low=0, high=3, shape=(stack_size, board_height, board_width), dtype='uint8')

        # Previous view
        self.previous_view = deque(maxlen=stack_size)

    # Observation
    def observation(self, _) -> NDArray:

        # Get Food Position
        food_pos = self.env.food.position

        # Get Snake's Head Position
        snake_head_pos = self.env.snake.head

        # Get Snake's Body Position
        snake_body_pos = self.env.snake.body

        # Create any np array of the same shape as the observation
        # and fill it with zeros
        current_view = np.zeros(self.observation_space.shape[1:], dtype='uint8')

        # Set the food position to 3
        current_view[food_pos.y, food_pos.x] = 3

        # Set the snake's head position to 1
        try:
            current_view[snake_head_pos.y, snake_head_pos.x] = 2
        except IndexError:
            # Means snake has crossed the boundary
            pass

        # Set the snake's body position to 1
        for pos in snake_body_pos:
            current_view[pos.y, pos.x] = 1

        # if previous observations then stack them
        if len(self.previous_view) < 1:
            self.previous_view.append(current_view)

        # Append current view to previous view
        self.previous_view.append(current_view)

        # Stack the views
        return np.stack((self.previous_view))
