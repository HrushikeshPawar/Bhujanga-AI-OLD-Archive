"""
This will contain all the requried helper functions and classes
"""
# /bhujanga_ai/helper.py
import matplotlib.pyplot as plt
from IPython import display


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


# Plotting Function
def plot(scores, mean_scores, fpath, save_plot=False):

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, color='red')
    plt.plot(mean_scores, color='green')
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

    if save_plot:
        plt.savefig(fpath)
