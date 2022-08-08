from collections import deque
from .basesnake import BaseSnake
from helper import Point, Direction
import logging
import os
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


# BASE CLASS FOR FINDERS
class Finder:
    def __init__(self, snake: BaseSnake, end: Point, log: bool = False, debug: bool = False):
        self.snake = snake
        self.start = snake.head
        self.end = end
        self.logging = log
        self.debug = debug
        self.visited = list()
        self.path = {}

    def find_path(self):
        # Will be overridden by child classes
        pass

    def get_path_directions(self) -> list:

        # Starting from the end, backtrack the path
        current = self.end
        path = [current]
        while current != self.start:
            if current.parent is None:
                break
            current = current.parent
            path.append(current)

        # Convert path to list of directions
        directions = {}
        for i in range(len(path) - 1):
            directions[path[i + 1]] = path[i] - path[i + 1]

        if self.debug:
            logger.debug('Got directions')
        self.path = directions


# BREADTH FIRST SEARCH
class BFS_Finder(Finder):

    def get_neighbors(self, current: Point) -> list:

        # Board range
        pos = current.copy()
        x_range = range(0, self.snake.board_width)
        y_range = range(0, self.snake.board_height)

        # Get all neighbours
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        if current == self.snake.head:
            neighbors = [pos + direction for direction in directions if direction != -self.snake.direction]
        else:
            neighbors = [pos + direction for direction in directions]

        # Remove invalid neighbors
        allowed_neighbors = [x for x in neighbors]
        for point in neighbors:

            #  Remove if out of bounds
            if point.x not in x_range:
                allowed_neighbors.remove(point)

            elif point.y not in y_range:
                allowed_neighbors.remove(point)

            # Remove if it is snakes head
            elif point == self.snake.head:
                allowed_neighbors.remove(point)

            # Remove if it is in snakes body
            elif point in self.snake.body:
                allowed_neighbors.remove(point)

        if self.debug:
            logger.debug(f'Neighbor for point {current} : {allowed_neighbors}')

        return allowed_neighbors

    def find_path(self):

        # Initialize the queue and visited list
        self.queue = deque()
        self.visited = []

        # Mark the start node as visited and enqueue it
        self.queue.append(self.start)

        if self.debug:
            logger.debug(f'Initial queue - {self.queue}')

        while self.queue:

            # Dequeue a vertex from queue
            current: Point = self.queue.popleft()
            if self.debug:
                logger.debug(f'Current point: ({current.x}, {current.y})')

            # Get all adjacent vertices of the dequeued vertex
            if current not in self.visited:
                self.visited.append(current)

                # Get neighbours from grid
                if self.debug:
                    logger.debug(f'Getting neighbors for point: {current}')

                for neighbour in self.get_neighbors(current):
                    neighbour: Point
                    if neighbour not in self.visited:
                        neighbour.parent = current
                        self.queue.append(neighbour)

                        if neighbour == self.end:
                            self.end = neighbour

                            if self.debug:
                                logger.debug('Found path')

                            self.get_path_directions()
                            return

            if self.debug:
                logger.debug(f'While loop running - Total visited{len(self.visited)}')
