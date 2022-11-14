import settings


# The Game Class
class GAME:
    """The main class which initializes and plays the game"""

    def __init__(
        self,
        height          : int = settings.HEIGHT,
        width           : int = settings.WIDTH,
        random_init     : bool = False,
        # agent           : BaseSnake = BaseSnake,
        # log             : bool = LOGGING,
        # debug           : bool = DEBUG,
        # show_display    : bool = PYGAME,
        save_gif        : bool = False,
    ) -> None:
        """Initialize the game"""

        # Initialize the game's environment (board)
        self.board_width    = width
        self.board_height   = height
        self.save_gif       = save_gif
        self.random_init    = random_init

        # Initialize the game's agent
        # self.agent = agent(height, width, random_init, log, debug)

        # Initialize the game's logging
        # self.logging = log
        # self.debug  = debug
        # self.logger = Setup_Logging()

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

        with contextlib.suppress(IndexError):
            for point in list(self.agent.body)[:-1]:
                pygame.draw.rect(self.display, BLUE, (point.x * BLOCKSIZE, point.y * BLOCKSIZE, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
                pygame.draw.rect(self.display, BLUE2, (point.x * BLOCKSIZE + 4, point.y * BLOCKSIZE + 4, 12 - BORDER, 12 - BORDER))

            # Drawing the tail
            point = list(self.agent.body)[-1]
            pygame.draw.rect(self.display, GREEN, (point.x * BLOCKSIZE, point.y * BLOCKSIZE, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
            pygame.draw.rect(self.display, GREEN2, (point.x * BLOCKSIZE + 4, point.y * BLOCKSIZE + 4, 12 - BORDER, 12 - BORDER))

        # Draw the score
        text = font.render(f'Score: {self.agent.score}', True, WHITE)
        self.display.blit(text, (1, 1))

        # Update the display
        pygame.display.update()

    # Printing the game details
    def __str__(self):
        """Print the game details"""
        # details = '\n\nGame Details\n\n'
        # details += f'Board Size: {str(self.board_width)}x{str(self.board_height)}' + '\n'
        # details += f'Snake Details: {str(self.agent)}'
        # if self.show_display:
        #     details += '\nDrawing Engine: PyGame\n'
        # else:
        #     details += '\nDrawing Engine: None\n'
        # return details
        if self.debug:
            self.logger.debug(f'Agent Name: {self.agent.__name__}')
            self.logger.debug(f'Board Sirze: {self.board_width}x{self.board_height}')
            self.logger.debug(f'Epochs: {EPOCHS}')
            self.logger.debug(f'Hidden Layers: {"x".join(str(x) for x in HIDDEN_LAYER_SIZES)}')
        return f'{self.agent.__name__} [{self.board_width}x{self.board_height}] [{EPOCHS}] [{"x".join(str(x) for x in HIDDEN_LAYER_SIZES)}]'
