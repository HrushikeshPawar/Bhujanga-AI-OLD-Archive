"""The main game class for the snake game"""
# /bhujanga_ai/game.py


# Import the required modules
import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP
import logging
import os
import pygame
import configparser
from time import perf_counter

# Import various helper and agent classes
from snakes.basesnake import BaseSnake
from snakes.pathfinding_snakes import BFS_Basic_Snake, BFS_LookAhead_Snake
from helper import BodyCollisionError, Direction, WallCollisionError


# Setup Config File
config = configparser.ConfigParser()
config.read(r'bhujanga_ai\settings.ini')

# Required Constants
B_HEIGHT = int(config['GAME - BASIC']['HEIGHT'])
B_WIDTH = int(config['GAME - BASIC']['WIDTH'])
CURSES = config['GAME - BASIC'].getboolean('CURSES')
PYGAME = not CURSES
LOGGING = config['GAME - BASIC'].getboolean('LOGGING')
DEBUG = config['GAME - BASIC'].getboolean('DEBUG')
LAP_TIME = int(config['GAME - BASIC']['LAP_TIME'])


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


# Setup for Curses
if CURSES:
    TIMEOUT = int(config['CURSES']['TIMEOUT'])
    HEAD_CHR = config['CURSES']['HEAD_CHR']
    BODY_CHR = config['CURSES']['BODY_CHR']
    TAIL_CHR = config['CURSES']['TAIL_CHR']
    FOOD_CHR = config['CURSES']['FOOD_CHR']


# Setup for Pygame
if PYGAME:

    # Initialize the pygame library
    pygame.init()
    font = pygame.font.Font(config['PYGAME']['FONT'], 20)

    # Define Constants
    BLOCKSIZE   = int(config['PYGAME']['BLOCKSIZE'])
    SPEED       = int(config['PYGAME']['SPEED'])
    BORDER      = int(config['PYGAME']['BORDER'])

    # Colors
    BLACK       = tuple(map(int, config['PYGAME']['BLACK'].split()))
    GREY        = tuple(map(int, config['PYGAME']['GREY'].split()))
    WHITE       = tuple(map(int, config['PYGAME']['WHITE'].split()))
    RED         = tuple(map(int, config['PYGAME']['RED'].split()))
    GREEN       = tuple(map(int, config['PYGAME']['GREEN'].split()))
    GREEN2      = tuple(map(int, config['PYGAME']['GREEN2'].split()))
    BLUE        = tuple(map(int, config['PYGAME']['BLUE'].split()))
    BLUE2       = tuple(map(int, config['PYGAME']['BLUE2'].split()))

    # DIRS
    IMG_DIR     = config['PYGAME']['IMG_DIR']


# The Game Class
class Game:
    """The main class which initializes and plays the game"""

    def __init__(
        self,
        height : int = B_HEIGHT,
        width : int = B_WIDTH,
        random_init : bool = False,
        agent : BaseSnake = BaseSnake,
        log : bool = LOGGING,
        debug : bool = DEBUG,
        # board : Union(curses.newwin, pygame.display) = None,
        board : curses.newwin or pygame.display = None,
    ) -> None:
        """Initialize the game"""

        # Initialize the game's environment (board)
        self.board_width = width
        self.board_height = height

        # Initialize the game's agent
        self.agent = agent(height, width, random_init, log, debug)
        self.agent : BaseSnake

        # Initialize the game's logging
        self.logging = log
        self.debug = debug

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

        # Initialize the curses board
        if CURSES:
            # Set the game speed
            self.board.timeout(TIMEOUT)
            self.timeout = TIMEOUT

            # Activate keypad mode
            self.board.keypad(1)

            # Draw the game board border
            self.board.border(0)

        # Initialize the pygame board
        if PYGAME:

            # Set the game display
            self.display = pygame.display.set_mode((self.board_width * BLOCKSIZE, self.board_height * BLOCKSIZE))

            # Set the game caption
            pygame.display.set_caption('Bhujanga-AI - Snake Game Solver')

            # Game clock
            self.clock = pygame.time.Clock()

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

    # Rendering the Game Board using pygame
    def render_pygame(self) -> None:
        """
        Update the UI
        """
        # Clear the screen
        self.display.fill(BLACK)

        # Draw the food
        pygame.draw.rect(self.display, RED, (self.agent.food.x * BLOCKSIZE, self.agent.food.y * BLOCKSIZE, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))

        # Draw the snake
        pygame.draw.rect(self.display, GREY, (self.agent.head.x * BLOCKSIZE, self.agent.head.y * BLOCKSIZE, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
        pygame.draw.rect(self.display, WHITE, (self.agent.head.x * BLOCKSIZE + 4, self.agent.head.y * BLOCKSIZE + 4, 12 - BORDER, 12 - BORDER))
        try:
            for point in list(self.agent.body)[:-1]:
                pygame.draw.rect(self.display, BLUE, (point.x * BLOCKSIZE, point.y * BLOCKSIZE, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
                pygame.draw.rect(self.display, BLUE2, (point.x * BLOCKSIZE + 4, point.y * BLOCKSIZE + 4, 12 - BORDER, 12 - BORDER))
            point = list(self.agent.body)[-1]
            pygame.draw.rect(self.display, GREEN, (point.x * BLOCKSIZE, point.y * BLOCKSIZE, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
            pygame.draw.rect(self.display, GREEN2, (point.x * BLOCKSIZE + 4, point.y * BLOCKSIZE + 4, 12 - BORDER, 12 - BORDER))
        except IndexError:
            pass

        # Draw the score
        text = font.render(f'Score: {self.agent.score}', True, WHITE)
        self.display.blit(text, (1, 1))

        # Update the display
        pygame.display.update()

    # Printing the game details
    def __str__(self):
        """Print the game details"""
        details = '\n\nGame Details\n\n'

        # Board Size
        details += 'Board Size: ' + str(self.board_width) + 'x' + str(self.board_height) + '\n'

        # Snake details
        details += 'Snake Details: ' + str(self.agent)

        # Drawing engine
        if CURSES:
            details += '\nDrawing Engine: Curses\n'
        elif PYGAME:
            details += '\nDrawing Engine: PyGame\n'

        return details


# The Main Initialization Function
def main():

    agent = agents[1]

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

        # Initialize the Game
        game = Game(agent=agent, board=board)
        if game.logging:
            logger.info('Curses is selected as the drawing engine')
            logger.info("Game has been initialized")

    # Initialize the Pygame Drawing Engine
    if PYGAME:

        # Initialize the Game
        game = Game(random_init=False, agent=agent, log=LOGGING)
        if game.logging:
            logger.info('PyGame is selected as the drawing engine')
            logger.info("Game has been initialized")

    if game.logging:
        logger.info(str(game))

    # The main game loop
    lap_time = perf_counter()
    score = 0
    try:
        while True:

            if CURSES:
                # Clear the screen and render the game board
                game.board.clear()
                game.board.border(0)

                # Render the game board
                try:
                    game.render_curses()
                except Exception as e:
                    logger.error(e)
                    pass

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
                if not game.agent.finder.path_exists():

                    if game.debug:
                        logger.debug("Finding the path")
                        logger.debug(f"Start: {(game.agent.head.x, game.agent.head.y)}, Goal: {game.agent.food.x, game.agent.food.y}")

                    # Set the finder to the current state of the game
                    game.agent.finder.start = game.agent.head
                    game.agent.finder.end = game.agent.food

                    # Find the path
                    game.agent.find_path()

                    if not game.agent.finder.path_exists():
                        if game.debug:
                            logger.debug('Path Not found!')
                        break
                    else:
                        if game.debug:
                            logger.debug(f"Path found from {game.agent.finder.start} to {game.agent.finder.end}- " + str(game.agent.finder.path))

                else:
                    direction = game.agent.finder.path[game.agent.head]

                    if game.debug:
                        logger.debug(f'Moving from {game.agent.head} in direction: {direction}')
                    game.agent.finder.path.pop(game.agent.head)
                    game.agent.move(direction)

                if PYGAME:
                    pygame.event.get()
                    game.render_pygame()
                    game.clock.tick(SPEED)

            except WallCollisionError:
                print("Wall Collision Error")
                break

            except BodyCollisionError:
                print("Body Collision Error")
                break

            if game.agent.score > score:
                score = game.agent.score
                lap_time = perf_counter()

            if perf_counter() - lap_time > LAP_TIME:
                raise KeyboardInterrupt

        # End the curses session
        if CURSES:
            curses.endwin()

        if game.logging:
            logger.info(f'Game Over - Your score is: {game.agent.score}')

    except KeyboardInterrupt:
        if CURSES:
            curses.endwin()
        if game.logging:
            logger.info(f'Game Over - Your score is: {game.agent.score}')


# Run the main function
if __name__ == "__main__":
    logger = Setup_Logging()
    agents = [BFS_Basic_Snake, BFS_LookAhead_Snake]
    main()
