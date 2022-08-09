from configparser import ConfigParser
import logging
import os
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob

# Import various helper functions and agents
from snakes.basesnake import BaseSnake
from snakes.pathfinding_snakes import BFS_Basic_Snake, BFS_LookAhead_Snake, BFS_LookAhead_LongerPath_Snake
from game import Game


# Setup Config File
config = ConfigParser()
config.read(r'bhujanga_ai\settings.ini')


# Required Constants
B_HEIGHT = int(config['GAME - BASIC']['HEIGHT'])
B_WIDTH = int(config['GAME - BASIC']['WIDTH'])
CURSES = config['GAME - BASIC'].getboolean('CURSES')
PYGAME = not CURSES
LOGGING = config['GAME - BASIC'].getboolean('LOGGING')
DEBUG = config['GAME - BASIC'].getboolean('DEBUG')
LAP_TIME = int(config['GAME - BASIC']['LAP_TIME'])
MEDIA_DIR = config['GAME - BASIC']['MEDIA_DIR']

# PYGAME Setup
# Initialize the pygame library
# pygame.init()
# font = pygame.font.Font(config['PYGAME']['FONT'], 20)

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


def Setup_Logging():
    # Setting up the logger
    logger = logging.getLogger('Game_Test_Runs')
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


def setup_agents():
    return [BFS_Basic_Snake, BFS_LookAhead_Snake, BFS_LookAhead_LongerPath_Snake]


def play_game(game):
    game.play()
    return game.score


def main():

    # setup Logging
    logger = Setup_Logging()

    # Setup the agents
    agents = setup_agents()

    # Storing score agent wise
    scores = {}

    # Games Info
    rounds = 10
    logger.info('Today we will test the following agents:')
    for agent in agents:
        logger.info(f'\t- {agent.__name__}')
    logger.info(f'Each agent will play {rounds} games.')
    print()

    # Setup the game
    for agent in agents:

        agent: BaseSnake
        print()
        logger.info(f'Starting Game for {agent.__name__}')
        scores[agent.__name__] = {}

        with ProcessPoolExecutor(max_workers=16) as executor:
            games = [Game(height=B_HEIGHT, width=B_WIDTH, agent=agent, log=False) for _ in range(rounds)]

            # Submit the jobs to the executor
            futures = [executor.submit(play_game, game) for game in games]

            # Getting the results
            results = [future.result() for future in tqdm(as_completed(futures), total=len(futures), desc=f'{agent.__name__} Games')]
        for i, result in enumerate(results):
            scores[agent.__name__][i + 1] = result
            logger.info(f'Score of Game {i+1}/{rounds} is {result}')

    # Store Scores
    json_path = os.path.join(MEDIA_DIR, r'Scores.json')
    with open(json_path, 'w') as jsonfile:
        json.dump(scores, jsonfile)

    # Print Scores
    print()
    for agent in agents:
        logger.info(f'Agent: {agent.__name__}')
        logger.info(f'\tTotal Games: {len(scores[agent.__name__])}')
        logger.info(f'\tAverage Score: {sum(scores[agent.__name__].values()) / len(scores[agent.__name__])}')
        logger.info(f'\tMax Score: {max(scores[agent.__name__].values())}')
        print()

    # Plot Scores
    cnt = len(glob(os.path.join(MEDIA_DIR, 'Campare Runs', r'Score - Test Run - *.png')))
    title = f'Compare Run {cnt + 1}'
    line_plot(json_path, name=os.path.join(MEDIA_DIR, 'Campare Runs', f'Score - Test Run - {cnt + 1}.png'), title=title)


# Line plot with mean line in seaborn
def line_plot(json_file, name, max_score=99, title=None):
    with open(json_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data)

    df.index = range(1, df.shape[0] + 1)
    # for col in df.columns[:-1]:
    sns.lineplot(data=df)
    plt.xlabel('Games')
    plt.ylabel(f'Score (Max Score - {max_score})')
    if title:
        plt.title(title)
    plt.savefig(name)


if __name__ == '__main__':
    # main()
    json_path = r'''C:\Users\hrush\OneDrive - iitgn.ac.in\Desktop\Projects\Bhujanga-AI\Media\Scores.json'''
    cnt = len(glob(os.path.join(MEDIA_DIR, 'Campare Runs', r'Score - Test Run - *.png')))
    title = f'Compare Run {cnt + 1}'
    line_plot(json_path, name=os.path.join(MEDIA_DIR, 'Campare Runs', f'Score - Test Run - {cnt + 1}.png'), title=title)
