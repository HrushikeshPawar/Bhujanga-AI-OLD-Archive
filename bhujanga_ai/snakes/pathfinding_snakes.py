from copy import deepcopy
from .basesnake import BaseSnake
from .utils import BFS_Finder
import os
import logging
import configparser

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


# BREADTH FIRST SEARCH (BFS - Basic) Snake
class BFS_Basic_Snake(BaseSnake):

    def __init__(self, height, width, random_init=False,  log : bool = False, debug : bool = False):
        super().__init__(height, width, random_init,  log, debug)
        self.finder = BFS_Finder(self, self.food, self.logging, self.debug)

    # Find the path to the food
    # This may look redundant for this snake
    # But it is in line with the preparation of framework for more advance snakes
    def find_path(self):

        self.finder.find_path()
        # return self.finder.get_path_directions()

    def move_snake(self):
        directions = self.finder.find_path()
        try:
            for direction in directions:
                self.move(direction)
                print(len(directions))
        except TypeError:
            pass

    def __str__(self):
        details = 'Basic BFS Snake\n'

        # Print snake's body
        details += f'Initial Snake Head: {self.head}\n'
        details += f'Initial Snake Direction: {self.direction}\n'

        # Print food
        details += f'Initial Food Place: {self.food}'

        return details

    __repr__ = __str__


# BREADTH FIRST SEARCH (BFS - Look Ahead) Snake
class BFS_LookAhead_Snake(BFS_Basic_Snake):

    def __init__(self, height, width, random_init=False,  log : bool = False, debug : bool = False):
        super().__init__(height, width, random_init,  log, debug)
        self.finder = BFS_Finder(self, self.food, self.logging, self.debug)

    def find_path(self):

        # When flag 0, follow all steps
        # When flag 1, jump to step 4
        flag = 0

        # Step 1: Find the path to the food using BFS
        # If the path exists, then move to step 2
        # If path doesnot exist, then move to step 4
        self.finder.find_path()

        # If the length of the snake is 1 i.e has no tail, return the path directly
        if len(self.body) == 0:
            return

        if not self.finder.path_exists():
            if self.finder.debug:
                logger.debug('Direct path from head to food does not exist')
            flag = 1
        else:
            if self.finder.debug:
                logger.debug(f'Found a direct path from {self.head} to {self.food}')

        # Step 2: Create a virtual snake and move it along the path
        # If the virtual snake can reach the tail of the original snake, then move to step 3
        # If the virtual snake cannot reach the tail of the original snake, then move to step 4
        if flag == 0:
            if self.finder.debug:
                logger.debug(f'Position of Original Snake: Head - {self.head}, Tail - {self.body[-1]}')
                logger.debug('Creating and moving a virtual snake')

            # virtual_snake = self.copy()  # copy the snake object
            virtual_snake = deepcopy(self)  # copy the snake object

            # Move the virtual snake along the path
            # virtual_snake.finder = self.finder.copy()
            virtual_snake.finder = deepcopy(self.finder)
            while virtual_snake.finder.path:
                direction = virtual_snake.finder.path[virtual_snake.head]
                virtual_snake.finder.path.pop(virtual_snake.head)
                virtual_snake.move(direction)

            # Now, check if the path to the tail of the virtual snake is reachable
            virtual_snake.finder = BFS_Finder(virtual_snake, virtual_snake.tail, self.logging, self.debug)
            virtual_snake.finder.start = virtual_snake.head
            virtual_snake.finder.find_path(exclude_tail=True)

            if self.finder.debug:
                logger.debug(f'Position of Virtual Snake: Head - {virtual_snake.head}, Tail - {virtual_snake.tail}')
                logger.debug(f'Virtual snake path finding path from {virtual_snake.finder.start} to {virtual_snake.finder.end}')
                logger.debug(f'Virtual snake path exists: {virtual_snake.finder.path_exists()}')

            if not virtual_snake.finder.path_exists():
                if self.finder.debug:
                    logger.debug('Virtual snake did not find a path to its tail')
                flag = 1
            else:
                if self.finder.debug:
                    logger.debug('Virtual snake found a path to its tail')

        # Step 3: Move the snake to the tail of the original snake
        if flag == 0:
            # for direction in self.finder.path:
            #     self.move(direction)
            return

        # Step 4: If no path exists, then follow the tail of the original snake
        if flag == 1:
            self.finder.end = self.tail.copy()
            self.finder.start = self.head.copy()

            if self.finder.debug:
                logger.debug('Following the tail of the original snake')
                logger.debug(f'Start(Head): {self.finder.start}, Goal(Tail): {self.finder.end}, Food Place: {self.food}')
            self.finder.find_path(exclude_tail=True)

    def move_snake(self):
        directions = self.finder.find_path()
        try:
            for direction in directions:
                self.move(direction)
                print(len(directions))
        except TypeError:
            pass

    def __str__(self):
        details = 'Look Ahead BFS Snake\n'

        # Print snake's body
        details += f'Initial Snake Head: {self.head}\n'
        details += f'Initial Snake Direction: {self.direction}\n'

        # Print food
        details += f'Initial Food Place: {self.food}'

        return details

    __repr__ = __str__
