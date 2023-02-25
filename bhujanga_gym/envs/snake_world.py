# bhujanga_gym\envs\snake_world.py


# Module leve imports
from .objects import Snake, Food
from .utils import Point, Direction, WallCollisionError, BodyCollisionError, TruncationError
from .renderer import SnakeGameRenderer
import bhujanga_gym.settings as settings

import gymnasium as gym
from gymnasium import spaces, Env
import os
import logging
import configparser
from datetime import datetime
from typing import List, Tuple, Dict, Union, Optional, Any
import pygame
import contextlib
import numpy as np

# Setup Config File
config = configparser.ConfigParser()
config.read(r'bhujanga_gym\settings.ini')


# Setup Logging
def Setup_Logging():
    # Setting up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] : %(name)s : %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    LOGS_DIR    = config['LOGGING']['LOG_DIR']
    name        = datetime.today().strftime("%Y-%m-%d")
    LOG_PATH    = os.path.join(LOGS_DIR, f'{name}_info.log')
    DEBUG_PATH  = os.path.join(LOGS_DIR, f'{name}_debug.log')

    # Check if file exists or create one
    if not os.path.exists(LOG_PATH):
        open(LOG_PATH, 'w').close()

    if not os.path.exists(DEBUG_PATH):
        open(DEBUG_PATH, 'w').close()

    file_handler_LOG = logging.FileHandler(LOG_PATH)
    file_handler_LOG.setLevel(logging.INFO)
    file_handler_LOG.setFormatter(formatter)

    file_handler_DEBUG = logging.FileHandler(DEBUG_PATH)
    file_handler_DEBUG.setLevel(logging.DEBUG)
    file_handler_DEBUG.setFormatter(formatter)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)

    logger.addHandler(file_handler_LOG)
    logger.addHandler(file_handler_DEBUG)
    # logger.addHandler(stream)

    return logger


# Define the logger
logger = Setup_Logging()


# The Snake World Environment
class SnakeWorldEnv(Env):

    metadata: Dict[str, Any] = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    # The Constructor
    def __init__(
        self,
        board_width: int = settings.BOARD_WIDTH,
        board_height: int = settings.BOARD_HEIGHT,
        random_init: bool = False,
        seed: Optional[int] = None,
        render_mode: str = None
    ):

        # Initialize the Env
        super().__init__()

        # # Set the seed
        # if seed:
        #     self.seed(seed)

        # Initialize the board
        self.board_width = board_width
        self.board_height = board_height
        self.total_points_to_earn = self.board_width * self.board_height - 2

        # Initialize the snake
        self.snake = Snake(board_width, board_height, random_init=random_init)

        # Initialize the food
        self.food = Food(board_width, board_height)

        # Initialize the action space
        # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        self.action_space = spaces.Discrete(4)

        # Initialize the observation space
        # By default the snake can only look up, down, left, right and see if the body or food is there
        # Each direction will have a 3 element array:
        # (UP, RIGHT, DOWN, LEFT)
        # (0, 0) -> Nothing
        # (1, 0) -> Food
        # (0, 1) -> Body
        # ((has_food, has_body), (has_food, has_body), (has_food, has_body), (has_food, has_body))
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(20,),
            dtype='int8'
        )

        # Action to Direction Mapping
        self.action_to_direction = {
            0: Direction.UP,
            1: Direction.RIGHT,
            2: Direction.DOWN,
            3: Direction.LEFT
        }

        # Direction to Action Mapping
        self.direction_to_action = {val: key for key, val in self.action_to_direction.items()}

        # Initialize the render mode
        assert render_mode in self.metadata['render_modes'] or render_mode is None, f'Invalid render mode: {render_mode}'
        self.render_mode = render_mode
        if render_mode == 'human':
            self.renderer = SnakeGameRenderer(self.board_width, self.board_height)

    # Place the food on the board
    def _place_food(self):
        # self.food.position = None
        logger.debug('Placing new food')
        logger.debug(f'Old position food at: {self.food.position}')
        self.food._place_food()
        while self.food.position is None or self.food.position in self.snake.body or self.food.position == self.snake.head:
            self.food._place_food()

        logger.debug(f'New position food at: {self.food.position}')

    # Reset the environment
    def reset(self, seed: Union[None, int] = None, options=None) -> Tuple[np.ndarray, dict]:

        # Set the seed
        super().reset(seed=seed)

        # Reset the snake
        self.snake.reset()
        self.move_count = 0

        # Reset the food
        self._place_food()

        # Return the observation
        return self._get_observation()

    # Get the observation
    def _get_observation(self) -> Tuple[np.ndarray, dict]:
        # Get the observation
        observation = []

        # Extend the observation for the UP direction
        observation.extend(self._get_observation_for_direction(Direction.UP))

        # Extend the observation for the RIGHT direction
        observation.extend(self._get_observation_for_direction(Direction.RIGHT))

        # Extend the observation for the DOWN direction
        observation.extend(self._get_observation_for_direction(Direction.DOWN))

        # Extend the observation for the LEFT direction
        observation.extend(self._get_observation_for_direction(Direction.LEFT))

        # Add the direction the snake is moving in
        observation.extend([
            self.snake.direction == Direction.UP,
            self.snake.direction == Direction.RIGHT,
            self.snake.direction == Direction.DOWN,
            self.snake.direction == Direction.LEFT
        ])

        # Add the food position relative to the snake's head
        observation.extend([
            self.food.position.x < self.snake.head.x,
            self.food.position.x > self.snake.head.x,
            self.food.position.y < self.snake.head.y,
            self.food.position.y > self.snake.head.y
        ])

        if self.render_mode == "human":
            self._render_frame()

        return np.asarray(observation, dtype=np.int8), {}  # info

    # Get the observation in a particular direction
    def _get_observation_for_direction(self, direction: Direction) -> List[int]:

        # Location of snake's head
        point = self.snake.head

        # Move that point in the given direction
        point = point.move(direction)

        # Check if the point is in the body
        if point in self.snake.body:
            return [1, 0, 0]
        # Check if the point is the food
        elif point == self.food.position:
            return [0, 1, 0]
        # Check if the point is out of bounds
        elif point.x not in range(self.board_width) or point.y not in range(self.board_height):
            return [0, 0, 1]
        # Else it is nothing
        else:
            return [0, 0, 0]

    # Step the environment
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Get the direction from the action
        direction = self.action_to_direction[action]

        # Info dict
        info = {
            'collision': False,
            'truncated': False
        }

        # Move the snake and get reward
        try:
            reward = self._move_snake(direction)
        except BodyCollisionError:
            self.snake.is_alive = False
            reward = -1
            info['collision'] = True
        except WallCollisionError:
            self.snake.is_alive = False
            reward = -1
            info['collision'] = True

        # The output has to be a tuple
        # (observation, reward, terminated, truncated, info)

        # Get the observation
        observation, _ = self._get_observation()

        # Check if the snake is dead
        terminated = not self.snake.is_alive

        # If the move count is greater than the max moves, truncate the episode
        if self.move_count > self.board_height * self.board_width:
            logger.debug(f'Truncating episode. The snake has made {self.move_count} moves')
            reward = -1
            truncated = True
            info['truncated'] = True

        elif len(self.snake) == self.board_height * self.board_width - 1:
            logger.debug(f'Truncating episode. The snake has made {self.move_count} moves')
            reward = 100
            truncated = True

        else:
            truncated = False

        return observation, reward, terminated, truncated, info

    # Move the snake in a particular direction
    def _move_snake(self, direction: Direction):

        # Check if given direction is from Direction class
        if not isinstance(direction, Point):
            raise TypeError("Direction must be from Direction class")

        # Check if the given direction is valid
        if direction not in self.direction_to_action.keys():
            raise ValueError("Direction must be one of the following: UP, DOWN, LEFT, RIGHT")

        # # Check if the given direction is not the opposite of the current direction
        # elif direction == -self.snake.direction:
        #     raise BodyCollisionError("Direction must be different from the opposite of the current direction")

        else:
            # Update the snake's direction
            # Change direction only when the given direction is not opposite of the current direction
            if direction != -self.snake.direction:
                self.snake.direction = direction

            # Update the snake's head position
            self.snake.body.appendleft(self.snake.head)
            self.snake.head = self.snake.head + self.snake.direction

            # Check for any collisions (with walls, snake's body or food)
            # Update the snake's body accordingly
            reward = self._check_collisions()

            # Update the move count
            self.move_count += 1

            return reward

    # Check for any collisions (with walls, snake's body or food)
    def _check_collisions(self) -> None:
        """Check for any collisions (with walls, snake's body or food)"""

        # Check if the snake's head is out of bounds
        if self.snake.head.x < 0 or self.snake.head.x >= self.board_width or self.snake.head.y < 0 or self.snake.head.y >= self.board_height:
            logger.debug("Snake collided with the wall")
            raise WallCollisionError

        # Check if the snake's head is on the food
        if self.snake.head == self.food.position:
            # Place the food at random location on the board
            logger.debug("Snake ate the food")
            self.snake.score += 1
            self.snake.tail: Point = self.snake.body.pop()
            self.snake.body.append(self.snake.tail.copy())

            # Reset the move count
            self.move_count = 0

            # Place the food at random location on the board
            logger.debug("Placing the food at random location on the board")
            self._place_food()
            return 1

        else:
            # Update the snake's body accordingly
            self.snake.body.pop()
            if len(self.snake.body) > 0:
                self.snake.tail: Point = self.snake.body.pop()
                self.snake.body.append(self.snake.tail.copy())

        # Check if the snake's head is on the snake's body
        # We check this after updating the snake's body
        # Otherwise it may wrongly detect the snake's head as on the snake's body
        # Specifically the tail point
        if self.snake.head in self.snake.body or self.snake.head == self.snake.tail:
            logger.debug("Body Collision Detected")
            logger.debug(f"Head Position: {self.snake.head}")
            logger.debug(f"Body Position: {self.snake.body}")
            logger.debug(f"Tail Position: {self.snake.tail}")
            raise BodyCollisionError
        else:
            return 0

    # Rendering the Game Board using pygame
    def render(self) -> None:
        """
        Update the UI
        """
        if self.render_mode == "human":
            self._render_frame()

    # Render the frame
    def _render_frame(self) -> None:

        # Clear the screen
        self.renderer.display.fill(settings.BLACK)

        # Draw the food
        pygame.draw.rect(
            self.renderer.display,
            settings.RED,
            (
                self.food.position.x * self.renderer.block_size,
                self.food.position.y * self.renderer.block_size,
                self.renderer.block_size - settings.BORDER,
                self.renderer.block_size - settings.BORDER
            )
        )

        # Draw the snake
        pygame.draw.rect(
            self.renderer.display,
            settings.GREY,
            (
                self.snake.head.x * self.renderer.block_size,
                self.snake.head.y * self.renderer.block_size,
                self.renderer.block_size - settings.BORDER,
                self.renderer.block_size - settings.BORDER
            )
        )
        pygame.draw.rect(
            self.renderer.display,
            settings.WHITE,
            (
                self.snake.head.x * self.renderer.block_size + 4,
                self.snake.head.y * self.renderer.block_size + 4,
                12 - settings.BORDER,
                12 - settings.BORDER
            )
        )

        with contextlib.suppress(IndexError):
            for point in list(self.snake.body)[:-1]:

                point: Point

                pygame.draw.rect(
                    self.renderer.display,
                    settings.BLUE,
                    (
                        point.x * self.renderer.block_size,
                        point.y * self.renderer.block_size,
                        self.renderer.block_size - settings.BORDER,
                        self.renderer.block_size - settings.BORDER
                    )
                )
                pygame.draw.rect(
                    self.renderer.display,
                    settings.BLUE2,
                    (
                        point.x * self.renderer.block_size + 4,
                        point.y * self.renderer.block_size + 4,
                        12 - settings.BORDER,
                        12 - settings.BORDER
                    )
                )

            # Drawing the tail
            point = list(self.snake.body)[-1]
            pygame.draw.rect(
                self.renderer.display,
                settings.GREEN,
                (
                    point.x * self.renderer.block_size,
                    point.y * self.renderer.block_size,
                    self.renderer.block_size - settings.BORDER,
                    self.renderer.block_size - settings.BORDER
                )
            )
            pygame.draw.rect(
                self.renderer.display,
                settings.GREEN2,
                (
                    point.x * self.renderer.block_size + 4,
                    point.y * self.renderer.block_size + 4,
                    12 - settings.BORDER,
                    12 - settings.BORDER
                )
            )

        # Draw the score
        text = self.renderer.font.render(f'Score: {self.snake.score}', True, settings.WHITE)
        self.renderer.display.blit(text, (1, 1))

        # Update the display
        pygame.display.update()

    # Close the game
    def close(self) -> None:
        """
        Close the game
        """
        pygame.display.quit()
        pygame.quit()
