"""This defines the base Snake Class which all other snakes will inherit from"""
# /bhujanga_ai/snakes/BaseSnake.py


# Import the required modules
from random import randint, sample
from collections import deque
from copy import deepcopy
import os
import logging
import configparser


# Import Helper Classes
from helper import Point, Direction, WallCollisionError, BodyCollisionError


# Required Constants
DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


# Setup Config File
config = configparser.ConfigParser()
config.read(r'bhujanga_ai\settings.ini')


def Setup_Logging():
    # Setting up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] : %(name)s : %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    LOG_PATH = config['LOGGING']['LOGGING_PATH']
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


# The BaseSnake Class
class BaseSnake:
    """The base snake class"""

    def __init__(self, height : int, width : int, random_init : bool = False, log : bool = False, debug : bool = False) -> None:

        # Initialize the snake's environment (board)
        self.board_width = width
        self.board_height = height
        self.logging = log
        self.debug = debug

        # Initialize the snake's initial position (snake's head)
        # Here we can take two approaches:
        # 1. Random Initialization
        # 2. Fixed Initialization
        self._initialize_snake(random_init)

        # Place the food at random location on the board
        # I first thought of placing the food at random location as well as fixed location on the board
        # But then I decided to just place it randomly
        self._place_food()

        # Initialize the snake's score
        self.score = len(self.body)

    # Initialize the snake (snake's head location)
    def _initialize_snake(self, random_init : bool) -> None:
        """Initialize the snake at random location or fixed location (center of board)"""

        # Initialize the snake's head at random location
        if random_init:
            self.head = Point(randint(0, self.board_width - 1), randint(0, self.board_height - 1))
            self.direction = sample(DIRECTIONS, 1)[0]

        # Initialize the snake's head at fixed location (center of board moving right)
        else:
            self.head = Point(self.board_width // 2, self.board_height // 2)
            self.direction = Direction.RIGHT

        # Here we use `deque` to store the snake's body
        # It has faster appending and popping operations compared to list
        # self.body = deque([self.head])
        # Thoughts: Not storing head in the body
        self.body = deque()
        self.tail = None

    # Place the food at random location on the board
    def _place_food(self) -> None:
        """Place the food at random location on the board"""

        # Place the food at random location
        # But check if the food is on the snake's body
        # If so, then place the food at random location again
        while True:
            self.food = Point(randint(0, self.board_width - 1), randint(0, self.board_height - 1))
            if self.food not in self.body and self.food != self.head:
                break

    # Move the snake in given direction
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
        elif direction == -self.direction:
            raise ValueError("Direction must be different from the opposite of the current direction")
            pass

        else:
            # Update the snake's direction
            # print('Moving in direction:', direction)
            self.direction = direction

            # Update the snake's head position
            self.body.appendleft(self.head)
            self.head = self.head + self.direction

            # Check for any collisions (with walls, snake's body or food)
            # Update the snake's body accordingly
            self._check_collisions()

    # Check for any collisions (with walls, snake's body or food)
    def _check_collisions(self) -> None:
        """Check for any collisions (with walls, snake's body or food)"""

        # Check if the snake's head is out of bounds
        if self.head.x < 0 or self.head.x > self.board_width - 1 or self.head.y < 0 or self.head.y > self.board_height - 1:
            # raise ValueError("Snake's head is out of bounds")
            raise WallCollisionError

        # Check if the snake's head is on the food
        if self.head == self.food:
            # Place the food at random location on the board
            self._place_food()
            self.score += 1
            self.tail = self.body.pop()
            self.body.append(self.tail.copy())
        else:
            # Update the snake's body accordingly
            self.body.pop()
            if len(self.body) > 0:
                self.tail = self.body.pop()
                self.body.append(self.tail.copy())

        # Check if the snake's head is on the snake's body
        # We check this after updating the snake's body
        # Otherwise it may wrongly detect the snake's head as on the snake's body
        # Specifically the tail point
        if self.head in self.body or self.head == self.tail:
            # raise ValueError("Snake's head is on the snake's body")
            # if self.debug:
            print(f'Debug is {self.debug}')
            logger.debug("Body Collision Detected")
            logger.debug(f"Head Position: {self.head}")
            logger.debug(f"Body Position: {self.body}")
            logger.debug(f"Tail Position: {self.tail}")
            raise BodyCollisionError

    # Printing the Snake Object
    def __str__(self) -> str:
        return f'''Snake(\n\thead\t  = {self.head},\n\tbody\t  = {self.body},\n\tdirection =   {self.direction}\n)'''

    # Make a copy of the snake object
    def copy(self) -> 'BaseSnake':
        # Creates a deepcopy of the snake
        agent = deepcopy(self)
        agent.finder = None
        return agent
