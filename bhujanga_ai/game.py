"""The main game class for the snake game"""
# /bhujanga_ai/game.py


# Import the required modules
import logging
import os
import configparser
from time import perf_counter

# Import various helper and agent classes
from snakes.basesnake import BaseSnake
from snakes.pathfinding_snakes import BFS_Basic_Snake, BFS_LookAhead_Snake, BFS_LookAhead_LongerPath_Snake
from helper import BodyCollisionError, WallCollisionError


# Setup Config File
config = configparser.ConfigParser()
config.read(r'bhujanga_ai\settings.ini')

# Required Constants
B_HEIGHT = int(config['GAME - BASIC']['HEIGHT'])
B_WIDTH = int(config['GAME - BASIC']['WIDTH'])
PYGAME = config['GAME - BASIC'].getboolean('PYGAME')
LOGGING = config['GAME - BASIC'].getboolean('LOGGING')
DEBUG = config['GAME - BASIC'].getboolean('DEBUG')
LAP_TIME = int(config['GAME - BASIC']['LAP_TIME'])
MEDIA_DIR = config['GAME - BASIC']['MEDIA_DIR']


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


logger = Setup_Logging()


# Setup for Pygame
if PYGAME:
    import pygame

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
        show_display : bool = True,
    ) -> None:
        """Initialize the game"""

        # Initialize the game's environment (board)
        self.board_width = width
        self.board_height = height
        self.show_display = show_display

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

        # Initialize the pygame board
        if show_display:

            # Set the game display
            self.display = pygame.display.set_mode((self.board_width * BLOCKSIZE, self.board_height * BLOCKSIZE))

            # Set the game caption
            pygame.display.set_caption('Bhujanga-AI - Snake Game Solver')

            # Game clock
            self.clock = pygame.time.Clock()

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

        if self.show_display:
            details += '\nDrawing Engine: PyGame\n'
        else:
            details += '\nDrawing Engine: None\n'

        return details

    # Game play
    def play(self) -> None:
        """Play the game"""

        if self.logging:
            logger.info(str(self))

        # The main game loop
        lap_time = perf_counter()
        score = 0
        try:
            while True:

                # Check if the user pressed the arrow key
                # If yes then update the direction of the snake
                # Check for collisions
                try:

                    # Move the snake
                    if not self.agent.finder.path_exists():

                        if self.debug:
                            logger.debug("Finding the path")
                            logger.debug(f"Start: {(self.agent.head.x, self.agent.head.y)}, Goal: {self.agent.food.x, self.agent.food.y}")

                        # Set the finder to the current state of the game
                        self.agent.finder.start = self.agent.head
                        self.agent.finder.end = self.agent.food

                        # Find the path
                        self.agent.find_path()

                        if not self.agent.finder.path_exists():
                            if self.debug:
                                logger.debug('Path Not found!')
                            break
                        else:
                            if self.debug:
                                logger.debug(f"Path found from {self.agent.finder.start} to {self.agent.finder.end}- " + str(self.agent.finder.path))

                    else:
                        direction = self.agent.finder.path[self.agent.head]

                        if self.debug:
                            logger.debug(f'Moving from {self.agent.head} in direction: {direction}')
                        self.agent.finder.path.pop(self.agent.head)
                        self.agent.move(direction)

                    if PYGAME:
                        pygame.event.get()
                        self.render_pygame()
                        self.clock.tick(SPEED)

                    if self.agent.score == 98:
                        self.agent.score += 1
                        raise KeyboardInterrupt

                except WallCollisionError:
                    print("Wall Collision Error")
                    break

                except BodyCollisionError:
                    print("Body Collision Error")
                    break

                if self.agent.score > score:
                    score = self.agent.score
                    lap_time = perf_counter()

                if perf_counter() - lap_time > LAP_TIME:
                    if self.logging:
                        logger.info('Lap Time Exceeded!')
                    raise KeyboardInterrupt

            if self.logging:
                logger.info(f'Game Over - Your score is: {self.agent.score}')

        except KeyboardInterrupt:
            if self.logging:
                logger.info(f'Game Over - Your score is: {self.agent.score}')

        # Final score
        self.score = self.agent.score
        return self.score


def Initialize_Game(agent):

    # Initialize the Pygame Drawing Engine
    if PYGAME:
        # Initialize the Game
        game = Game(random_init=False, agent=agent, log=LOGGING, debug=DEBUG, show_display=True)
        if game.logging:
            logger.info('PyGame is selected as the drawing engine')
            logger.info("Game has been initialized")

    # Initialize the Game with no display
    else:
        game = Game(random_init=False, agent=agent, log=LOGGING, debug=DEBUG, show_display=False)
        if game.logging:
            logger.info('No Drawing engine is selected')
            logger.info("Game has been initialized")

    return game


# Run the main function
if __name__ == "__main__":
    agents = [BFS_Basic_Snake, BFS_LookAhead_Snake, BFS_LookAhead_LongerPath_Snake]
    Ggame = Initialize_Game(agents[2])
    Ggame.play()
