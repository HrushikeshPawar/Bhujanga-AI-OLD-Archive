from .basesnake import BaseSnake
from .utils import BFS_Finder


# BREADTH FIRST SEARCH (BFS) Snake
class BFS_Basic_Snake(BaseSnake):

    def __init__(self, height, width, random_init=False):
        super().__init__(height, width, random_init)
        self.finder = BFS_Finder(self, self.food)

    def move_snake(self):
        directions = self.finder.find_path()
        try:
            for direction in directions:
                self.move(direction)
                print(len(directions))
        except TypeError:
            pass
