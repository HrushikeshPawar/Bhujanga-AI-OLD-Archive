"""
This will contain all the requried helper functions and classes
"""
# /bhujanga_ai/helper.py
from copy import deepcopy
from collections import deque
import logging
from datetime import datetime
import os
from bhujanga_gym.envs.snake_world import SnakeWorldEnv


def Setup_Logging():
    # Setting up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] : %(name)s : %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    LOGS_DIR    = r'.\logs'
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


logger = Setup_Logging()


# Error Classes
# Wall Collision Error
class WallCollisionError(Exception):
    """Raised when the snake collides with the wall"""
    pass


# Collision with body Error
class BodyCollisionError(Exception):
    """Raised when the snake collides with its body"""
    pass


# Point Class
class Point:
    """Class to help in defining the location of Snake's head and body and also the food"""

    # Changing it from namedtuple to normal class
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

    # Adding and Subtracting Points
    # Used in changing the direction of the snake
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    # Negating the point coordinates
    def __neg__(self) -> 'Point':
        return Point(-self.x, -self.y)

    # Multiplying the Point by a scalar(int) from both the sides (left and right multiplication)
    def __mul__(self, other: int) -> 'Point':
        return Point(self.x * other, self.y * other)

    def __rmul__(self, other: int) -> 'Point':
        return Point(self.x * other, self.y * other)

    # The way point object is printed on screen
    def __str__(self) -> str:
        return f"Point({self.x}, {self.y})"

    __repr__ = __str__

    # Checking if the point is equal to another point
    def __eq__(self, other: 'Point') -> bool:
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    # Checking if the point is not equal to another point
    def __ne__(self, other: 'Point') -> bool:
        return not isinstance(other, Point) or self.x != other.x or self.y != other.y

    # Creats the hash of the point object
    # Don't know why I added this
    # If I don't find any use then I will remove it
    # TODO: Find out how this can be used
    def __hash__(self) -> int:
        return hash((self.x, self.y))

    # Copying the point object
    def copy(self) -> 'Point':
        return Point(self.x, self.y)

    # Distance function
    def distance(self, other: 'Point') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    # Move in give direction
    def move(self, direction: 'Direction') -> 'Point':
        return self + direction


# Direction Class
class Direction:
    """Class to help in defining the direction of Snake"""

    # Four Major Directions
    UP    = Point(0, -1)
    DOWN  = Point(0, 1)
    LEFT  = Point(-1, 0)
    RIGHT = Point(1, 0)

    # Four Diagonal Directions
    UP_LEFT    = Point(-1, -1)
    UP_RIGHT   = Point(1, -1)
    DOWN_LEFT  = Point(-1, 1)
    DOWN_RIGHT = Point(1, 1)

    # Eight Sub Diagonal Directions
    UP_UP_LEFT       = Point(-1, -2)
    UP_UP_RIGHT      = Point(1, -2)
    UP_LEFT_LEFT     = Point(-2, -1)
    UP_RIGHT_RIGHT   = Point(2, -1)
    DOWN_DOWN_LEFT   = Point(-1, 2)
    DOWN_DOWN_RIGHT  = Point(1, 2)
    DOWN_LEFT_LEFT   = Point(-2, 1)
    DOWN_RIGHT_RIGHT = Point(2, 1)

    # Multiplication of a point object by a scalar
    def __mul__(self, other: int) -> Point:
        return Point(self.value.x * other, self.value.y * other)

    # Multiplication of a scalar by a point object
    def __rmul__(self, other: int) -> Point:
        return Point(self.value.x * other, self.value.y * other)


# BASE CLASS FOR FINDERS
class Finder:
    def __init__(self, env: SnakeWorldEnv):
        self.env = env
        self.start: Point = self.env.snake.head
        self.end: Point = self.env.snake.tail
        self.visited = []
        self.path = {}

    def find_path(self):
        # Will be overridden by child classes
        pass

    def get_path_directions(self) -> list:
        # sourcery skip: de-morgan, dict-comprehension, merge-else-if-into-elif, simplify-len-comparison, swap-if-else-branches, use-named-expression, while-guard-to-condition

        # Starting from the end, backtrack the path
        current = self.end
        path = [current]
        while current != self.start:
            if current.parent is None:
                break
            current = current.parent
            path.append(current)

        if self.start not in path:
            path.append(self.start)
        logger.debug(f"Path from {self.start} to {self.end}: {path[::-1]}")

        # Convert path to list of directions
        directions = {}
        for i in range(len(path) - 1):
            directions[path[i + 1]] = path[i] - path[i + 1]

        # Check if path is empty
        if len(directions) == 0:
            logger.debug('Path is empty')
        else:
            logger.debug('Got directions')
            self.path = directions

    def copy(self) -> 'Finder':
        # Create a deep copy of the finder
        return deepcopy(self)

    def path_exists(self) -> bool:
        return len(self.path) > 0 if self.path else False


# BREADTH FIRST SEARCH
class BFS_Finder(Finder):

    def get_neighbors(self, current: Point, exclude_tail: bool = False) -> list:

        # Board range
        pos = current.copy()
        x_range = range(self.env.snake.board_width)
        y_range = range(self.env.snake.board_height)

        # Get all neighbours
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        if current == self.env.snake.head:
            neighbors = [pos + direction for direction in directions if direction != -self.env.snake.direction]
        else:
            neighbors = [pos + direction for direction in directions]

        # Remove invalid neighbors
        allowed_neighbors = deepcopy(neighbors)
        for point in neighbors:

            #  Remove if out of bounds
            if point.x not in x_range or point.y not in y_range or point == self.env.snake.head:
                allowed_neighbors.remove(point)

            # Remove if it is in snakes body
            elif exclude_tail:
                body = deepcopy(self.env.snake.body)
                body.pop()
                if point in body:
                    allowed_neighbors.remove(point)

            elif point in self.env.snake.body:
                allowed_neighbors.remove(point)

        return allowed_neighbors

    def find_path(self, start: Point, end: Point, exclude_tail: bool = True) -> None:

        self.start = start
        self.end = end

        logger.debug('Starting BFS')
        logger.debug(f'Finding path from Start : {self.start} to End : {self.end}')
        logger.debug(f'Excluding Tail : {exclude_tail}')
        logger.debug(f'Walls are : {list(self.env.snake.body)[:-1]}')

        # Initialize the queue and visited list
        self.queue = deque()
        self.visited = []
        self.path = {}

        # Mark the start node as visited and enqueue it
        self.queue.append(self.start)

        # if self.debug:
        #     logger.debug(f'Initial queue - {self.queue}')

        while self.queue:

            # Dequeue a vertex from queue
            current: Point = self.queue.popleft()
            # if self.debug:
            #     logger.debug(f'Current point: ({current.x}, {current.y})')

            # Get all adjacent vertices of the dequeued vertex
            if current not in self.visited:
                self.visited.append(current)

                # Get neighbours from grid
                # if self.debug:
                #     logger.debug(f'Getting neighbors for point: {current}')

                for neighbour in self.get_neighbors(current, exclude_tail):
                    neighbour: Point
                    if neighbour not in self.visited:
                        neighbour.parent = current
                        self.queue.append(neighbour)

                        if neighbour == self.end:
                            self.end.parent = current

                            logger.debug('Found path')

                            self.get_path_directions()
                            return

        logger.debug('No path found')
