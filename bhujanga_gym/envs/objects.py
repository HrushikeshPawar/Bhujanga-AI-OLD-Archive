# bhujanga_gym\envs\snake.py

# Import the required modules
from random import randint, sample
from collections import deque
from copy import deepcopy
import os
import logging
import configparser
from datetime import datetime


# Import Helper Classes
from .utils import Point, Direction, WallCollisionError, BodyCollisionError, TruncationError


# Required Constants
DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


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
    logger.addHandler(stream)

    return logger


# Define the logger
logger = Setup_Logging()


# The Snake Class
class Snake:

    # The Constructor
    def __init__(
        self,
        board_width: int,
        board_height: int,
        random_init: bool = False,
    ):

        # Setup the board
        self.board_width = board_width
        self.board_height = board_height

        # Setup the snake
        self.random_init = random_init
        self.reset()

    # Setup the snake
    def __setup_snake(self):
        """Initialize the snake at random location or fixed location (center of board)"""

        # Initialize the snake's head at random location
        if self.random_init:
            logger.debug('Initializing the snake at random location')
            self.head = Point(randint(0, self.board_width - 1), randint(0, self.board_height - 1))
            self.direction = sample(DIRECTIONS, 1)[0]

        # Initialize the snake's head at fixed location (center of board moving right)
        else:
            logger.debug('Initializing the snake at fixed location, center of board moving right')
            self.head = Point(self.board_width // 2, self.board_height // 2)
            self.direction = Direction.RIGHT

        # Here we use `deque` to store the snake's body
        # It has faster appending and popping operations compared to list
        # self.body = deque([self.head])
        # Thoughts: Not storing head in the body
        self.body = deque()
        self.tail = None

        logger.info(f'Snake initialized at {self.head} moving {self.direction}')

    # Reset the snake
    def reset(self) -> None:
        # Initialize the snake's initial position (snake's head)
        # Here we can take two approaches:
        # 1. Random Initialization
        # 2. Fixed Initialization
        logger.info('Resetting the snake')
        self.__setup_snake()

        # Initialize the snake's score
        self.score      = len(self.body)
        self.is_alive   = True

    # Printing the Snake Object
    def __str__(self) -> str:
        return f'''Snake(\n\thead\t  = {self.head},\n\tbody\t  = {self.body},\n\tdirection =   {self.direction}\n)'''

    # Make a copy of the snake object
    def copy(self) -> 'Snake':
        # Creates a deepcopy of the snake
        agent = deepcopy(self)
        return agent


# The Food Class
class Food:

    # The Constructor
    def __init__(self, board_width: int, board_height: int):
        self.board_width = board_width
        self.board_height = board_height
        self.position = None

    # Place the food at random location on the board
    def _place_food(self) -> None:
        self.position = Point(randint(0, self.board_width - 1), randint(0, self.board_height - 1))
