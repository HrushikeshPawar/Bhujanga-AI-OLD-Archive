"""The main game class for the snake game"""
# /bhujanga_ai/game.py


# Import the required modules
import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP
import logging
import os
import pygame
# from datetime import datetime

# Import various helper and agent classes
from snakes.basesnake import BaseSnake
from snakes.pathfinding_snakes import BFS_Basic_Snake
from helper import BodyCollisionError, Direction, WallCollisionError


# Required Constants
B_HEIGHT = 10
B_WIDTH = 10
CURSES = True
PYGAME = not CURSES
LOGGING = True


# Required Dicts
key_to_direction = {
    KEY_LEFT: Direction.LEFT,
    KEY_RIGHT: Direction.RIGHT,
    KEY_UP: Direction.UP,
    KEY_DOWN: Direction.DOWN
}


def Setup_Logging():
    # Setting up the logger
    logger = logging.getLogger('game')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] : %(name)s : %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    LOG_PATH = os.path.join(r"""C:\Users\hrush\OneDrive - iitgn.ac.in\Desktop\Projects\Bhujanga-AI""", 'Logs.log')

    # Check if file exists or create one
    if not os.path.exists(LOG_PATH):
        open(LOG_PATH, 'w').close()

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream)

    return logger


# Setup for Curses
if CURSES:
    TIMEOUT = 10  # The speed of the game (In curses, the timeout is in milliseconds, higher is slower)
    HEAD_CHR = 'H'
    BODY_CHR = '#'
    TAIL_CHR = 'T'
    FOOD_CHR = 'O'


# Setup for Pygame
if PYGAME:

    # Initialize the pygame library
    pygame.init()
    font = pygame.font.Font('Lora-Regular.ttf', 20)

    # Define Constants
    BLOCKSIZE   = 20
    SPEED       = 20
    BORDER      = 3

    # Colors
    BLACK       = (0, 0, 0)
    GREY        = (150, 150, 150)
    WHITE       = (255, 255, 255)
    RED         = (255, 0, 0)
    GREEN       = (0, 255, 0)
    GREEN2      = (100, 255, 0)
    BLUE        = (0, 0, 255)
    BLUE2       = (0, 100, 255)

    # DIRS
    IMG_DIR     = 'Pics'


# The Game Class
class Game:
    """The main class which initializes and plays the game"""

    def __init__(
        self,
        height : int,
        width : int,
        random_init : bool = False,
        agent : BaseSnake = BaseSnake,
        log : bool = False,
        board : curses.newwin = None,
    ) -> None:
        """Initialize the game"""

        # Initialize the game's environment (board)
        self.board_width = width
        self.board_height = height

        # Initialize the game's agent
        self.agent = agent(height, width, random_init)

        # Initialize the game's logging
        self.logging = log

        # Initialize the game's initial position (snake's head)
        # Here we can take two approaches:
        # 1. Random Initialization
        # 2. Fixed Initialization
        self.agent._initialize_snake(random_init)

        # Place the food at random location on the board
        # I first thought of placing the food at random location as well as fixed location on the board
        # But then I decided to just place it randomly
        self.agent._place_food()

        # Initialize the board
        self.board = board
        self.timeout = TIMEOUT

    # Rendering the Game Board on terminal using curses
    def render_curses(self) -> None:

        # Rendering the Head
        self.board.addstr(self.agent.head.y, self.agent.head.x, HEAD_CHR)

        # Rendering the Body
        if len(self.agent.body) > 1:
            for i in range(len(self.agent.body) - 1):
                self.board.addstr(self.agent.body[i].y, self.agent.body[i].x, BODY_CHR)

        # Rendering the Tail
        if len(self.agent.body) > 0:
            self.board.addstr(list(self.agent.body)[-1].y, list(self.agent.body)[-1].x, TAIL_CHR)
        # if self.agent.tail:
        #     self.board.addstr(self.agent.tail.y, self.agent.tail.x, TAIL_CHR)

        # Rendering the Food
        self.board.addstr(self.agent.food.y, self.agent.food.x, FOOD_CHR)

        # Render the score counter (for me it is the lenght counter)
        # self.board.addstr(0, 5, "Length: " + str(self.agent.score))
        self.board.addstr(0, 5, str(self.agent.score))

        # Refresh the board
        self.board.refresh()


# The Main Initialization Function
def main():

    # Initialize the Curse Drawing Engine
    if CURSES:

        # Initialize the curses library
        curses.initscr()

        # Disable the echoing of keys to the screen
        curses.noecho()

        # Make the cursor invisible
        curses.curs_set(0)

        # Create the game board window
        board = curses.newwin(B_HEIGHT, B_WIDTH, 0, 0)

        # Set the game speed
        board.timeout(TIMEOUT)

        # Activate keypad mode
        board.keypad(1)

        # Draw the game board border
        board.border(0)

        # Initialize the Game
        game = Game(B_HEIGHT, B_WIDTH, random_init=False, agent=BFS_Basic_Snake, log=LOGGING, board=board)
        if game.logging:
            logger.info('Curses is selected as the drawing engine')
            logger.info("Game has been initialized")

    # Initialize the Pygame Drawing Engine
    if PYGAME:

        # Initialize the Game
        game = Game(B_HEIGHT * BLOCKSIZE, B_WIDTH * BLOCKSIZE, random_init=False, agent=BFS_Basic_Snake, log=LOGGING, board=board)
        if game.logging:
            logger.info('PyGame is selected as the drawing engine')
            logger.info("Game has been initialized")

    # The main game loop
    while True:
        # print(time())

        # Clear the screen and render the game board
        game.board.clear()
        game.board.border(0)

        # Render the game board
        game.render_curses()

        # Get the key pressed by the user
        key = game.board.getch()

        # Check if the user pressed the exit key
        if key == ord('q'):
            break

        # Check if the user pressed the arrow key
        # If yes then update the direction of the snake
        # Check for collisions
        try:
            # if key in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN]:
            #     game.agent.move(key_to_direction[key])

            # # If no direction key is pressed, continue moving in the same direction
            # elif key == -1:
            #     # print(game.agent)
            #     game.agent.move(game.agent.direction)

            # Move the snake
            if game.agent.finder.path == {}:
                # logger.info("Finding the path")
                # logger.info(f"Start: {(game.agent.head.x, game.agent.head.y)}, Goal: {game.agent.food.x, game.agent.food.y}")
                game.agent.finder.start = game.agent.head
                game.agent.finder.end = game.agent.food
                game.agent.finder.find_path()
                # logger.info("Path found - " + str(game.agent.finder.path))
                if game.agent.finder.path == {}:
                    # logger.info('Path Not found!')
                    break
            else:
                direction = game.agent.finder.path[game.agent.head]
                # print('Got direction: ', direction)
                # logger.info('Moving in direction: ', direction)
                game.agent.finder.path.pop(game.agent.head)
                game.agent.move(direction)
                # logger.info('Path length: ', len(game.agent.finder.path))

        except WallCollisionError:
            print("Wall Collision Error")
            break

        except BodyCollisionError:
            print("Body Collision Error")
            break

    # End the curses session
    curses.endwin()
    print("Game Over")
    print('Your score is: ' + str(game.agent.score))


# Run the main function
if __name__ == "__main__":
    logger = Setup_Logging()
    main()
