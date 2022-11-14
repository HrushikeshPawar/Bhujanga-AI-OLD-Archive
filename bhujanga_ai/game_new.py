import os
import logging
from datetime import datetime
from typing import Union
import contextlib
from glob import glob1
from loguru import logger as lg
from time import perf_counter

# Torch Imports
import torch
from torch.utils.tensorboard.writer import SummaryWriter


# DONE: Constants
# Find a good way to store these constants and make them accessible to the rest of the program.
# Constants should be stored in a separate file.
# Constants should be in all caps.
from constants import LOGGING, DEBUG, LOGGING_PATH, LOG_DIR, DEBUG_PATH  # Logging Constant

# Constants for PYGAME
from constants import FONT, BLOCKSIZE, SPEED, BORDER, BLACK, GREY, WHITE, RED, GREEN, GREEN2, BLUE, BLUE2, PYGAME

# Game Constants
from constants import WIDTH, HEIGHT, COMPLETE_MODEL_DIR, CHECKPOINT_DIR, GIF_PATH, GRAPH_PATH

# RL Constants
from constants import EPSILON_MAX, EPOCHS, HIDDEN_LAYER_SIZES

# Import Snakes
from snakes.basesnake import BaseSnake
from snakes.pathfinding_snakes import BFS_Basic_Snake, BFS_LookAhead_Snake, BFS_LookAhead_LongerPath_Snake
from snakes.reinforced_snakes import DQN_Snake, Double_Q_Snake

# Helper Functions
from helper import Direction, WallCollisionError, BodyCollisionError, plot


TODAY       = datetime.now().strftime('%Y%m%d')

# Snake Classes and their Names
SNAKES = {
    BaseSnake.__name__  : BaseSnake,
    BFS_Basic_Snake.__name__: BFS_Basic_Snake,
    BFS_LookAhead_Snake.__name__: BFS_LookAhead_Snake,
    BFS_LookAhead_LongerPath_Snake.__name__: BFS_LookAhead_LongerPath_Snake,
    DQN_Snake.__name__: DQN_Snake
}


# DONE: Logging
# Logging should be done using the logging module.
# Setup the streams for debug, info, warning and error.
# Setup the format for the logs.
# Setup the log levels.
def Setup_Logging():
    # Setting up the logger
    logger = logging.getLogger('Game')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] : %(name)s : %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # LOG_PATH = config['LOGGING']['LOGGING_PATH']
    LOG_PATH = os.path.join(LOG_DIR, f'{TODAY}.log')

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


# Setting up Pygame
if PYGAME:
    import pygame

    # Initialize the pygame library
    pygame.init()
    font = pygame.font.Font(FONT, 20)


# The game class should be a generic class that can be used to create any game.
# The game class should have the following attributes:
#   1. The game board.
#   2. The game players.
#   3. The game state.
class GAME:
    """The main class which initializes and plays the game"""

    def __init__(
        self,
        height          : int = HEIGHT,
        width           : int = WIDTH,
        random_init     : bool = False,
        agent           : BaseSnake = BaseSnake,
        log             : bool = LOGGING,
        debug           : bool = DEBUG,
        show_display    : bool = PYGAME,
        save_gif        : bool = False,
    ) -> None:
        """Initialize the game"""

        # Initialize the game's environment (board)
        self.board_width    = width
        self.board_height   = height
        self.show_display   = show_display
        self.save_gif       = save_gif
        self.random_init    = random_init

        # Initialize the game's agent
        self.agent = agent(height, width, random_init, log, debug)

        # Initialize the game's logging
        self.logging = log
        self.debug  = debug
        self.logger = Setup_Logging()

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


# TODO: The Game (RL) Class
# The RL game class should inherit from the game class.
# It should initialize the player with its Deep Neural Networks.
class RLGame(GAME):

    @lg.catch
    def __init__(self, agent: DQN_Snake, random_start: bool = False, nth_try: Union[int, None] = None, show_display: bool = PYGAME) -> None:
        super().__init__(agent=agent, log=LOGGING, debug=DEBUG, show_display=show_display, random_init=random_start)

        # Required Constants
        self.record          = 0
        self.average_score   = 0
        self.total_reward    = 0
        self.current_epsilon = EPSILON_MAX
        nth_try              = nth_try if nth_try is not None else len(glob1(os.path.join(COMPLETE_MODEL_DIR, 'RL Models and Checkpoints'), "*.pth")) + 1
        self.title           = f'{str(self)} - Try {nth_try}'

        # Required Paths
        folder = 'RL Models and Checkpoints'
        self.CHK_FILE_PATH = os.path.join(CHECKPOINT_DIR, folder, f'{self.title}.pth')
        self.MODEL_FPATH   = os.path.join(COMPLETE_MODEL_DIR, folder, f'{self.title}.pth')

        # Check if any checkpoint exists
        if os.path.exists(self.CHK_FILE_PATH):
            self.plot_scores, self.plot_average_scores, self.total_score = self.load_checkpoint(self.CHK_FILE_PATH)
        else:
            self.plot_scores         = []
            self.plot_average_scores = []
            self.total_score         = 0

    # Play the given step (move in given direction and calculate the reward)
    def play_step(self, action: Union[Direction, None] = None) -> None:
        """
        Play a single step of the game
        """

        # Move the Snake and calculate the reward
        try:
            self.agent.move(action)
            game_over = False

            # Save this frame
            if self.save_gif:
                pygame.image.save(self.display, os.path.join(GIF_PATH, f"screenshot0{self.img_cnt}.png"))
                self.img_cnt += 1

        except WallCollisionError:
            game_over = True
            if self.save_gif:
                self._save_gif()

        except BodyCollisionError:
            game_over = True
            if self.save_gif:
                self._save_gif()

        # Step 5 - Update UI and clock
        if PYGAME:
            pygame.event.get()
            self.render_pygame()
            self.clock.tick(SPEED)

        if self.agent.score == 98:
            self.agent.score += 1
            raise KeyboardInterrupt

        # Step 6 - Return Game Over and Score
        if self.debug:
            self.logger.debug(f'|{game_over}|{self.agent.score}|{self.agent.reward}|')
        return game_over, self.agent.score, self.agent.reward

    # Load details from checkpoint file
    def load_checkpoint(self, checkpoint_file: str) -> tuple:
        checkpoint = torch.load(checkpoint_file)
        plot_scores         = checkpoint['scores']
        plot_average_scores = checkpoint['average_scores']
        total_score         = checkpoint['total_score']
        self.agent.games_played   = checkpoint['epoch']
        self.agent.epsilon  = checkpoint['epsilon']
        self.agent.q_net.load_state_dict(checkpoint['state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['trainer'])

        if checkpoint['target_dict'] == {}:
            self.agent.target_q_net = None
        else:
            self.agent.target_q_net.load_state_dict(checkpoint['target_dict'])

        return plot_scores, plot_average_scores, total_score

    # Decrease the epsilon value
    def epsilon_decrement(self) -> None:
        self.agent.epsilon = self.agent.epsilon_min if self.agent.epsilon <= self.agent.epsilon_min else self.agent.epsilon * self.agent.epsilon_decay

    # Process Done
    def process_done(self, score: int) -> None:
        # Reset Moves Count
        self.moves_count = 0

        # Reset the game if game is over
        self.agent.reset(self.random_init)

        # update the Epsilon
        self.epsilon_decrement()

        # Update the Number of Games Played
        self.agent.games_played += 1

        # Train the model
        start_time = perf_counter()
        self.agent.train_long_memory()
        if self.logging:
            self.logger.info(f'Time required for Train over sample: {perf_counter() - start_time:.5f}')

        # Create CheckPoint
        checkpoint = {
            'epoch': self.agent.games_played,
            'state_dict': self.agent.q_net.state_dict(),
            'target_dict': {} if self.agent.target_q_net is None else self.agent.target_q_net.state_dict(),
            'trainer': self.agent.optimizer.state_dict(),
            'scores': self.plot_scores,
            'average_scores': self.plot_average_scores,
            'total_score': self.total_score,
            'epsilon': self.current_epsilon
        }

        # Save the Checkpoint
        start_time = perf_counter()
        self.agent.q_net.save_checkPoints(state=checkpoint, checkpoint_name=self.CHK_FILE_PATH)
        if self.logging:
            self.logger.info(f'Time required for saving checkpoint: {perf_counter() - start_time:.5f}')

        # Update the target network
        if self.agent.target_q_net is not None:
            self.agent.target_updated_counter += 1
            if self.agent.target_updated_counter % self.agent.target_update_freq == 0:
                start_time = perf_counter()
                self.agent.target_q_net.load_state_dict(self.agent.q_net.state_dict())
                if self.logging:
                    self.logger.info(f'Time required for Updating Target Q Net: {perf_counter() - start_time:.5f}')

        # If the score is best so far, the save the model
        if score > self.record:
            self.record = score
            start_time = perf_counter()
            self.agent.q_net.save(self.MODEL_FPATH)
            if self.logging:
                self.logger.info(f'Time required for saving Model: {perf_counter() - start_time:.5f}')

        # Plotting
        self.plot_scores.append(score)
        self.total_score += score
        average_score = self.total_score / self.agent.games_played
        self.plot_average_scores.append(average_score)

        # Write to TensorBoard
        if TB_WRITE:
            self.write_summary(score, average_score)

        start_time = perf_counter()
        plot_fpath = os.path.join(GRAPH_PATH, f'{self.title}.png')
        plot(self.plot_scores, self.plot_average_scores, plot_fpath, self.title, save_plot=True)
        if self.logging:
            self.logger.info(f'Time required for Ploting Model: {perf_counter() - start_time:.5f}')

        return average_score

    # Summary Writer
    def write_summary(self, score, average_score):
        start_time = perf_counter()
        writer.add_scalar('Score', score, self.agent.games_played)
        writer.add_scalar('Average Score', average_score, self.agent.games_played)
        writer.add_scalar('Epsilon', self.agent.epsilon, self.agent.games_played)
        writer.add_scalar('Loss', self.agent.loss, self.agent.games_played)
        if self.logging:
            self.logger.info(f'Time required for Summary Writer: {perf_counter() - start_time:.5f}')

    # The Training Loop
    def train(self) -> None:

        self.agent : DQN_Snake

        # Start Training
        moves_count = 0
        total_moves = 0
        try:
            while self.agent.games_played <= EPOCHS:
                game_start_time = perf_counter()

                # Get Old State
                old_state = self.agent.get_state()

                # Get Action
                direction, idx = self.agent.get_direction(old_state)

                # Perform Action
                done, score, reward  = self.play_step(direction)
                new_state            = self.agent.get_state()
                self.total_reward   += reward
                self.current_epsilon = self.agent.epsilon

                # If reward is positive, reset the moves_count
                total_moves += 1
                moves_count = 0 if reward > 0 else moves_count + 1

                #  If moves_count is greater than or equal to 2 * width * height, then kill the agent
                if self.debug:
                    self.logger.debug(f'Move Count: {moves_count} | Condition: {moves_count >= 2 * self.board_width * self.board_height:}')
                if moves_count >= 2 * self.board_width * self.board_height:
                    moves_count = 0
                    done = True

                # Training on short memory
                # self.agent.train_short_memory(old_state, idx, reward, new_state, done)

                # Remember
                self.agent.remember(old_state, idx, reward, new_state, done)

                # Train on long memory
                if done:
                    if self.logging:
                        self.logger.info(f'Total Moves performed: {total_moves}')
                    total_moves = 0
                    done = False
                    average_score = self.process_done(score)

                    if self.logging:
                        self.logger.info(f'Time Required for Game Completion: {perf_counter() - game_start_time:.5f}')

                    self.logger.info(f'|Game: {self.agent.games_played: >3}|Score: {score: >2}|Average Score: {average_score:.3f}|Record: {self.record}|Total Reward: {self.total_reward: >4}|Epsilon: {self.agent.epsilon:.4f}|Loss: {self.agent.loss:.4f}|')
                    if self.logging:
                        print()

        except KeyboardInterrupt:
            if self.logging:
                self.logger.info(f'Game Over - Your score is: {self.agent.score}')


TB_WRITE = True
if __name__ == '__main__':

    # Agent
    agents = [BFS_Basic_Snake, BFS_LookAhead_Snake, BFS_LookAhead_LongerPath_Snake, DQN_Snake, Double_Q_Snake]

    # Game
    nth_try = 1
    game = RLGame(agents[-2], nth_try=nth_try)

    # Summary Writer
    if TB_WRITE:
        writer = SummaryWriter(f'TBRuns/{game.title}')

    # Train
    game.train()
