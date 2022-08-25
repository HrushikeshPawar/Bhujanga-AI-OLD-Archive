"""The main game class for the snake game"""
# /bhujanga_ai/game.py


# Import the required modules
import logging
import os
import configparser
from time import perf_counter
import torch
from glob import glob1
import contextlib
from torch.utils.tensorboard import SummaryWriter

# Import various helper and agent classes
from snakes.basesnake import BaseSnake
from snakes.pathfinding_snakes import BFS_Basic_Snake, BFS_LookAhead_Snake, BFS_LookAhead_LongerPath_Snake
from snakes.reinforced_snakes import BASIC_Q_Snake
from helper import BodyCollisionError, WallCollisionError, Direction, plot


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
COMPLETE_MODEL_DIR  = config['GAME - BASIC']['COMPLETE_MODEL_DIR']
CHECKPOINT_DIR      = config['GAME - BASIC']['CHECKPOINT_DIR']
LOG_DIR             = config['LOGGING']['LOG_DIR']
DEBUG_PATH          = config['LOGGING']['DEBUG_PATH']
EPOCHS              = int(config['RL SNAKE']['epochs'])
GIF_PATH            = os.path.join(MEDIA_DIR, 'GIFs')
GRAPH_PATH          = os.path.join(MEDIA_DIR, 'Graphs')


def Setup_Logging():
    # Setting up the logger
    logger = logging.getLogger('Game')
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
        save_gif : bool = False,
    ) -> None:
        """Initialize the game"""

        # Initialize the game's environment (board)
        self.board_width = width
        self.board_height = height
        self.show_display = show_display
        self.save_gif = save_gif
        self.random_init = random_init

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
        details = '\n\nGame Details\n\n'
        details += f'Board Size: {str(self.board_width)}x{str(self.board_height)}' + '\n'
        details += f'Snake Details: {str(self.agent)}'
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


# The Game class for RL Snake Game
class RLGame(Game):

    # Play the given step (move in given direction and calculate the reward)
    def play_step(self, action: Direction = None) -> None:
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

            return game_over, self.agent.score, self.agent.reward

        # Step 5 - Update UI and clock
        if PYGAME:
            pygame.event.get()
            self.render_pygame()
            self.clock.tick(SPEED)

        if self.agent.score == 98:
            self.agent.score += 1
            raise KeyboardInterrupt

        # Step 6 - Return Game Over and Score
        return game_over, self.agent.score, self.agent.reward

    # Required Variables for RL Snake Game
    def required(self, nth_model : int = None, nth_chk : int = None) -> tuple:
        record          = 0
        average_score   = 0
        total_reward    = 0
        model           = self.agent.__name__
        current_epsilon = 1

        if nth_model is None:
            nth_model = len(glob1(os.path.join(COMPLETE_MODEL_DIR, 'RL Models and Checkpoints'), "*.pth")) + 1

        if nth_chk is None:
            nth_chk = len(glob1(os.path.join(CHECKPOINT_DIR, 'RL Models and Checkpoints'), "*.pth")) + 1

        folder = 'RL Models and Checkpoints'
        board_size = f'{self.board_width} X {self.board_height}'
        CHK_FILE_PATH = os.path.join(CHECKPOINT_DIR, folder, f'{model} - Game {nth_chk} - Board {board_size}.pth')
        MODEL_FPATH   = os.path.join(COMPLETE_MODEL_DIR, folder, f'{model} - Game {nth_model} - Board {board_size}.pth')

        return record, average_score, total_reward, current_epsilon, MODEL_FPATH, CHK_FILE_PATH

    # Load details from checkpoint file
    def load_checkpoint(self, checkpoint_file: str) -> tuple:
        checkpoint = torch.load(checkpoint_file)
        plot_scores         = checkpoint['scores']
        plot_average_scores = checkpoint['average_scores']
        total_score         = checkpoint['total_score']
        self.agent.games_played   = checkpoint['epoch']
        self.agent.epsilon  = checkpoint['epsilon']
        self.agent.model.load_state_dict(checkpoint['state_dict'])
        self.agent.trainer.optimizer.load_state_dict(checkpoint['trainer'])

        return plot_scores, plot_average_scores, total_score

    # The Training Loop
    def train(self, nth_model : int = None, nth_chk : int = None) -> None:

        # Define required variables Initialize the logging of results
        record, average_score, total_reward, current_epsilon, MODEL_FPATH, CHK_FILE_PATH = self.required(nth_model, nth_chk)
        model = f'{self.agent.__name__} {n} (Without Short Train)'
        board_size = f'{self.board_width} X {self.board_height}'
        self.agent : BASIC_Q_Snake

        # Check if any checkpoint exists
        if os.path.exists(CHK_FILE_PATH):
            plot_scores, plot_average_scores, total_score = self.load_checkpoint(CHK_FILE_PATH)
        else:
            plot_scores         = []
            plot_average_scores = []
            total_score         = 0

        # Start Training
        moves_count = 0
        try:
            while self.agent.games_played <= EPOCHS:

                # Get Old State
                old_state = self.agent.get_state()

                # Get Action
                direction, idx = self.agent.get_direction(old_state)

                # Perform Action
                done, score, reward = self.play_step(direction)
                new_state           = self.agent.get_state()
                total_reward        += reward
                current_epsilon     = self.agent.epsilon

                # If reward is positive, reset the moves_count
                moves_count = 0 if reward > 0 else moves_count + 1

                #  If moves_count is greater than or equal to 2 * width * height, then kill the agent
                if moves_count >= 2 * self.board_width * self.board_height:
                    done = True

                if self.debug:
                    self._extracted_from_train_34(old_state, idx, direction, reward)

                # Training on short memory
                # self.agent.train_short_memory(old_state, idx, reward, new_state, done)

                # Remember
                self.agent.remember(old_state, idx, reward, new_state, done)

                # Train on long memory
                if done:

                    # Reset Moves Count
                    moves_count = 0

                    self.agent.reset(self.random_init)
                    self.agent.epsilon = self.agent.epsilon_min if self.agent.epsilon <= self.agent.epsilon_min else self.agent.epsilon * self.agent.epsilon_decay
                    self.agent.games_played += 1
                    self.agent.train_long_memory()

                    # Create CheckPoint
                    checkpoint = {
                        'epoch': self.agent.games_played,
                        'state_dict': self.agent.model.state_dict(),
                        'trainer': self.agent.trainer.optimizer.state_dict(),
                        'scores': plot_scores,
                        'average_scores': plot_average_scores,
                        'total_score': total_score,
                        'epsilon': current_epsilon
                    }

                    # Save the Checkpoint
                    self.agent.model.save_checkPoints(state=checkpoint, checkpoint_name=CHK_FILE_PATH)

                    # If the score is best so far, the save the model
                    if score > record:
                        record = score
                        self.agent.model.save(MODEL_FPATH)

                    # logger.info(f'|Game: {self.agent.games_played: <3}|Score: {score}|Record: {record: <4}|Total Reward: {total_reward}|Current Epsilon: {self.agent.epsilon:.4f}|Loss: {self.agent.trainer.loss:.4f}|')

                    # Plotting
                    plot_scores.append(score)
                    total_score += score
                    average_score = total_score / self.agent.games_played
                    plot_average_scores.append(average_score)

                    print(f'|Game: {self.agent.games_played: >3}|Score: {score: >2}|Average Score: {average_score:.3f}|Record: {record}|Total Reward: {total_reward: >4}|Current Epsilon: {self.agent.epsilon:.4f}|Loss: {self.agent.trainer.loss:.4f}|')

                    # Write to TensorBoard
                    writer.add_scalar('Score', score, self.agent.games_played)
                    writer.add_scalar('Average Score', average_score, self.agent.games_played)
                    writer.add_scalar('Epsilon', self.agent.epsilon, self.agent.games_played)
                    writer.add_scalar('Loss', self.agent.trainer.loss, self.agent.games_played)

                    plot_fpath = os.path.join(GRAPH_PATH, f'{model} - Game {nth_model} - Board {board_size}.png')
                    plot(plot_scores, plot_average_scores, plot_fpath, save_plot=True)

        except KeyboardInterrupt:
            if self.logging:
                logger.info(f'Game Over - Your score is: {self.agent.score}')

    # TODO Rename this here and in `train`
    def _extracted_from_train_34(self, old_state, idx, direction, reward):
        logger.debug(f"State: {old_state}")
        logger.debug(f"Prediction: {idx}")
        logger.debug(f"Action: {direction}")
        logger.debug(f"Reward: {reward}")
        logger.debug(f"Head: {self.agent.head}")
        # logger.debug(f"Food: {self.agent.food}")
        logger.debug("")


def Initialize_Game(agent, RL=False, random_start: bool = False) -> Game:

    # Initialize the Pygame Drawing Engine
    if PYGAME:
        # Initialize the Game
        if RL:
            game = RLGame(agent=agent, log=LOGGING, debug=DEBUG, show_display=True, random_init=random_start)
        else:
            game = Game(agent=agent, log=LOGGING, debug=DEBUG, show_display=True, random_init=random_start)
        if game.logging:
            logger.info('PyGame is selected as the drawing engine')
            logger.info("Game has been initialized")

    # Initialize the Game with no display
    else:
        if RL:
            game = RLGame(agent=agent, log=LOGGING, debug=DEBUG, show_display=False, random_init=random_start)
        else:
            game = Game(agent=agent, log=LOGGING, debug=DEBUG, show_display=False, random_init=random_start)
        if game.logging:
            logger.info('No Drawing engine is selected')
            logger.info("Game has been initialized")

    return game


# Run the main function
if __name__ == "__main__":
    agents = [BFS_Basic_Snake, BFS_LookAhead_Snake, BFS_LookAhead_LongerPath_Snake, BASIC_Q_Snake]
    Ggame = Initialize_Game(agents[-1], RL=True, random_start=False)
    n = 3

    writer = SummaryWriter(f'TBRuns/{Ggame.agent.__name__} {n}')
    # writer.add_graph(Ggame.agent.model, torch.zeros(1, 11))

    Ggame.train(n, n)
