from .basesnake import BaseSnake
from .utils import BFS_Finder


# BREADTH FIRST SEARCH (BFS) Snake
class BFS_Basic_Snake(BaseSnake):

    def __init__(self, height, width, random_init=False):
        super().__init__(height, width, random_init)
        self.finder = BFS_Finder(self, self.food, self.log)

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
