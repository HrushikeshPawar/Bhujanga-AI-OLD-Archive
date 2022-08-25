# Import required libraries
from configparser import ConfigParser
from collections import deque
import numpy as np
import random
import torch
import os
from datetime import datetime
from typing import Tuple


# Import the necessary classes and helper functions
from .basesnake import BaseSnake
from bhujanga_ai.snakes.basesnake import DIRECTIONS
from .utils import Q_Network_Basic, Q_Trainer_Basic
from helper import Direction, BodyCollisionError, WallCollisionError, Point
import logging


# Required Constants
# The model settings
config = ConfigParser()
config.read(r'bhujanga_ai\settings.ini')
MAX_MEMORY          = int(config['RL SNAKE']['max_memory'])
BATCH_SIZE          = int(config['RL SNAKE']['batch_size'])
LEARNING_RATE       = float(config['RL SNAKE']['learning_rate'])
DISCOUNT_RATE       = float(config['RL SNAKE']['discount_rate'])
EPSILON_MAX         = float(config['RL SNAKE']['epsilon_max'])
EPSILON_MIN         = float(config['RL SNAKE']['epsilon_min'])
EPSILON_DECAY       = float(config['RL SNAKE']['epsilon_decay'])
EPOCHS              = int(config['RL SNAKE']['epochs'])
HIDDEN_LAYER_SIZES  = list(map(int, (config['RL SNAKE']['hidden_layer_sizes']).split(', ')))
COMPLETE_MODEL_DIR  = config['GAME - BASIC']['COMPLETE_MODEL_DIR']
CHECKPOINT_DIR      = config['GAME - BASIC']['CHECKPOINT_DIR']
LOG_DIR             = config['LOGGING']['LOG_DIR']
DEBUG_PATH          = config['LOGGING']['DEBUG_PATH']
TODAY               = datetime.now().strftime('%Y%m%d')


def Setup_Logging():
    # Setting up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] : %(name)s : %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    LOG_PATH = os.path.join(LOG_DIR, f'{TODAY}.log')
    DEBUG_PATH = config['LOGGING']['DEBUG_PATH']

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
    logger.addHandler(stream)

    return logger


logger = Setup_Logging()


# BASIC Q-LEARNING REINFORCED SNAKE
class BASIC_Q_Snake(BaseSnake):

    __name__ = 'BASIC Q-Snake'

    def __init__(
        self,
        height: int,
        width: int,
        random_init: bool = False,
        log: bool = False,
        debug: bool = False
    ) -> None:
        super().__init__(height, width, random_init, log, debug)

        self.games_played            = 0
        self.memory             = deque(maxlen=MAX_MEMORY)
        self.learning_rate      = LEARNING_RATE
        self.discount_rate      = DISCOUNT_RATE
        self.epsilon            = EPSILON_MAX
        self.epsilon_min        = EPSILON_MIN
        self.epsilon_decay      = EPSILON_DECAY
        self.epochs             = EPOCHS
        self.hidden_layer_sizes = HIDDEN_LAYER_SIZES
        self.reward             = 0

        # Initialize the model
        self.model   = Q_Network_Basic(
            input_size=11,  # 4 directions * 6 features
            hidden_sizes=HIDDEN_LAYER_SIZES,
            output_size=3
        )  # Neural Network Model
        self.trainer = Q_Trainer_Basic(self.model, learning_rate=LEARNING_RATE, gamma=self.discount_rate)

    # Move the snake in given direction (Removed the reverse direction checking)
    def move(self, direction : Direction) -> None:
        """Move the snake in given direction"""

        # Check if given direction is from Direction class
        if not isinstance(direction, Point):
            raise TypeError("Direction must be from Direction class")
            # pass

        # Check if the given direction is valid
        if direction not in DIRECTIONS:
            raise ValueError("Direction must be one of the following: UP, DOWN, LEFT, RIGHT")
            # pass

        # Check if the given direction is not the opposite of the current direction
        # elif direction == -self.direction:
        #     raise ValueError("Direction must be different from the opposite of the current direction")

        # else:
        # Update the snake's direction
        # print('Moving in direction:', direction)
        self.direction = direction

        # Update the snake's head position
        self.body.appendleft(self.head)
        self.head = self.head + self.direction

        # Check for any collisions (with walls, snake's body or food)
        # Update the snake's body accordingly
        self._check_collisions()

    # Check for any collisions (with walls, snake's body or food) [Added rewards]
    def _check_collisions(self) -> None:
        """Check for any collisions (with walls, snake's body or food)"""

        # Check if the snake's head is out of bounds
        if self.head.x not in range(self.board_width) or self.head.y not in range(self.board_height):
            self.reward = -10  # self.board_height * self.board_width
            raise WallCollisionError

        # Check if the snake's head is on the food
        if self.head == self.food:
            # Place the food at random location on the board
            self._place_food()
            self.score += 1
            self.reward = 10
            self.tail = self.body.pop()
            self.body.append(self.tail.copy())
        else:
            # Update the snake's body accordingly
            # self.reward = -1 if self._distance_to_food_increased() else 2
            self.reward = 0
            self.body.pop()
            if len(self.body) > 0:
                self.tail = self.body.pop()
                self.body.append(self.tail.copy())

        # Check if the snake's head is on the snake's body
        # We check this after updating the snake's body
        # Otherwise it may wrongly detect the snake's head as on the snake's body
        # Specifically the tail point
        if self.head in self.body or self.head == self.tail:
            if self.debug:
                logger.debug("Body Collision Detected")
                logger.debug(f"Head Position: {self.head}")
                logger.debug(f"Body Position: {self.body}")
                logger.debug(f"Tail Position: {self.tail}")
            raise BodyCollisionError

    # Check if the distance has increased
    def _distance_to_food_increased(self) -> bool:
        """Check if the distance has increased"""

        current = self.head.copy()
        previous = self.body[0].copy()

        return current.distance(self.food) > previous.distance(self.food)

    # Check for collision with walls, snake's body or food
    def _check_collision_for_point(self, point : Point) -> bool:
        """Check for collision with walls, snake's body or food"""

        # Check if the given point is out of bounds
        if point.x not in range(self.board_width) or point.y not in range(self.board_height):
            return True

        # Check if the given point is on the snake's body
        return point in self.body

    # Get the vision in given direction
    def get_vision(self, direction: Direction):

        # What will the snake see in the given direction?
        # 1. Is the food visible?
        # 2. Is the wall visible?
        # 3. Is the snake's body visible?
        # 4. Distance to the food
        # 5. Distance to the wall
        # 6. Distance to the snake's body

        current = self.head.copy()
        FOOD = False
        WALL = False
        BODY = False
        DISTANCE_TO_FOOD = 0
        DISTANCE_TO_WALL = 0
        DISTANCE_TO_BODY = 0
        dist = 0

        # Current Settings
        # The Snake is able to see past food
        # But is not able to see past itself
        while True:
            current = current.move(direction)
            dist += 1

            if current.x not in range(self.board_width + 1) or current.y not in range(self.board_height + 1):
                WALL = True
                DISTANCE_TO_WALL = 1 / dist
                break
            elif current == self.food:
                FOOD = True
                DISTANCE_TO_FOOD = 1 / dist

            elif current in self.body:
                BODY = True
                DISTANCE_TO_BODY = 1 / dist
                break

        return [FOOD, WALL, BODY, DISTANCE_TO_FOOD, DISTANCE_TO_WALL, DISTANCE_TO_BODY]

    # Define the state of the game
    def get_state_old(self) -> np.array:
        # sourcery skip: assign-if-exp, boolean-if-exp-identity, remove-pass-body, remove-unnecessary-cast

        # BASIC:
        # Agent able to look in 4 directions
        # What will it see in each direction?
        # 1. Is the food visible?
        # 2. Is the wall visible?
        # 3. Is the snake's body visible?
        # 4. Distance to the food
        # 5. Distance to the wall
        # 6. Distance to the snake's body

        state = []

        # Get the current state of the game
        for dir in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            state.extend(self.get_vision(dir))

        return np.array(state, dtype=float)

    def get_state(self) -> np.array:

        head = self.head.copy()

        # Points near the snake head
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        # Bool value of current direction
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            (dir_l and self._check_collision_for_point(point_l)) or
            (dir_r and self._check_collision_for_point(point_r)) or
            (dir_u and self._check_collision_for_point(point_u)) or
            (dir_d and self._check_collision_for_point(point_d)),

            # Is Danger Right?
            (dir_l and self._check_collision_for_point(point_u)) or
            (dir_r and self._check_collision_for_point(point_d)) or
            (dir_u and self._check_collision_for_point(point_r)) or
            (dir_d and self._check_collision_for_point(point_l)),

            # Is Danger Left?
            (dir_l and self._check_collision_for_point(point_d)) or
            (dir_r and self._check_collision_for_point(point_u)) or
            (dir_u and self._check_collision_for_point(point_l)) or
            (dir_d and self._check_collision_for_point(point_r)),

            # Then add the direction of snake's head
            dir_u,
            dir_r,
            dir_d,
            dir_l,

            # Food Location
            self.food.x < self.head.x,   # Food is to the left of the snake
            self.food.y < self.head.y,   # Food is above the snake
            self.food.x > self.head.x,   # Food is to the right of the snake
            self.food.y > self.head.y    # Food is below the snake
        ]

        return np.array(state, dtype=float)

    # Remember the last state, action, reward, next state
    def remember(self, state: np.ndarray, action, reward: int, next_state: np.ndarray, done: bool) -> None:

        # We store the memory in the form of (state, action, reward, next_state, done)
        self.memory.append((state, action, reward, next_state, done))

    # Train the long term memory (after every episode - From a selected sample)
    def train_long_memory(self) -> None:

        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        # The stored memory is in the form of (state, action, reward, next_state, done) tuple
        # We convert it into a list of each element separately
        states, actions, rewards, next_states, dones = zip(*sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # Train the short term memory (after every step)
    def train_short_memory(self, state: np.ndarray, action: list, reward: int, next_state: np.ndarray, done: bool) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    # Choose the action to take
    def get_direction(self, state: np.ndarray) -> Tuple[Direction, int]:

        # Random Moves = Tradeoff between exploration and exploitation
        # if self.epsilon <= self.epsilon_min:
        #     self.epsilon = self.epsilon_min
        # self.epsilon = self.epsilon_min if self.epsilon <= self.epsilon_min else self.epsilon * self.epsilon_decay

        action = [0, 0, 0]  # [straight, right, left]

        if random.random() < self.epsilon:
            idx = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            idx = torch.argmax(prediction).item()

        action[idx] = 1
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        dir_id = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = self.direction  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_id = (dir_id + 1) % 4
            new_dir = clock_wise[next_id]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_id = (dir_id - 1) % 4
            new_dir = clock_wise[next_id]  # left turn r -> u -> l -> d

        return new_dir, idx
