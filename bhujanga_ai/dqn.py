# Required Libs
import logging
import os
import random
# Import settings from the settings.ini file
from configparser import ConfigParser
from datetime import datetime
import pygame
from typing import Optional
import json
from time import perf_counter

import numpy as np
# Import the torch libs
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from bhujanga_gym.envs.snake_world import SnakeWorldEnv
from bhujanga_gym.settings import BOARD_WIDTH, BOARD_HEIGHT

# Import the QNet and Replay Memory
from .utils import QNetwork, ReplayMemory, PrioritizedReplayMemory

config = ConfigParser()
config.read(os.path.join('bhujanga_ai', 'settings.ini'))


# Constants for the DQN Agent
try:
    SEED            = int(config['DQN']['seed'])
except Exception:
    SEED = None
MEMORY_CAPACITY = int(config['DQN']['max_memory'])
BATCH_SIZE      = int(config['DQN']['batch_size'])
LEARNING_RATE   = float(config['DQN']['learning_rate'])
DISCOUNT_FACTOR = float(config['DQN']['discount_factor'])
EPS_MAX         = float(config['DQN']['epsilon_max'])
EPS_MIN         = float(config['DQN']['epsilon_min'])
EPS_DECAY       = float(config['DQN']['epsilon_decay'])
TOTAL_GAMES     = int(config['DQN']['total_games'])
TARGET_UPDATE_FREQ  = int(config['DQN']['target_update_freq'])
HIDDEN_LAYER_SIZES  = [int(x) for x in config['DQN']['hidden_layer_sizes'].split(',')]
CHECK_POINT_FREQ    = int(config['DQN']['checkpoint_freq'])
ALPHA           = float(config['DQN']['alpha'])


# Setup Logging
def Setup_Logging():
    # Setting up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] : %(name)s : %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    LOGS_DIR    = r'.\logs'
    name        = datetime.today().strftime("%Y-%m-%d")
    LOG_PATH    = os.path.join(LOGS_DIR, f'{name}_info.log')
    DEBUG_PATH  = os.path.join(LOGS_DIR, f'{name}_debug.log')

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
    # logger.addHandler(stream)

    return logger


# Define the logger
logger = Setup_Logging()


# The DQN Agent Class
class DQNAgent:

    # Constructor
    def __init__(
        self,
        env: SnakeWorldEnv,
        seed: int = SEED,
        memory_capacity: int = MEMORY_CAPACITY,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        discount_factor: float = DISCOUNT_FACTOR,
        eps_max: float = EPS_MAX,
        eps_min: float = EPS_MIN,
        eps_decay: float = EPS_DECAY,
        total_games: int = TOTAL_GAMES,
        target_update_freq: int = TARGET_UPDATE_FREQ,
        hidden_layer_sizes: list = HIDDEN_LAYER_SIZES,
        addon: Optional[str] = 'basic'
    ):

        global MODEL_NAME
        MODEL_NAME = self.get_name(addon)

        # Set the seed
        self.seed = seed
        if seed:
            self.setup_seed()

        # Set the environment
        self.env = env

        # Set the replay memory
        self.memory = ReplayMemory(memory_capacity)

        # Set the Model
        self.model = QNetwork(
            input_size=self.env.observation_space.shape[0],
            output_size=self.env.action_space.n,
            hidden_layer_sizes=hidden_layer_sizes
        )

        # Set the Target Model
        self.target_model = QNetwork(
            input_size=self.env.observation_space.shape[0],
            output_size=self.env.action_space.n,
            hidden_layer_sizes=hidden_layer_sizes,
        )
        self.target_update_freq = target_update_freq
        self.update_target_model()

        # Set the hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.total_games = total_games
        self.steps_done = 0
        self.games_completed = 0

        # Set the optimizer
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        # Set the epsilon
        self.eps = self.eps_max

        # Set the loss function
        self.loss_fn = MSELoss().to(self.model.device)

        # if TBOARD:
        #     f_name = os.path.join('charts', 'dqn_per', MODEL_NAME)
        #     self.writter = SummaryWriter(f_name)

    # Function to set the universal seed
    def setup_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    # Function to create a name for the model
    def get_name(self, addon: Optional[str] = 'basic'):
        return f'[{datetime.now().strftime("%Y%m%d %H%M%S")}]-{BOARD_WIDTH}x{BOARD_HEIGHT}-{",".join([str(x) for x in HIDDEN_LAYER_SIZES])}-{BATCH_SIZE}-{LEARNING_RATE}-{DISCOUNT_FACTOR}-{TOTAL_GAMES}-{addon}'

    # Function to update the epsilon
    def update_epsilon(self):
        # self.eps = self.eps_min + (self.eps_max - self.eps_min) * np.exp(-1. * self.steps_done / self.eps_decay)
        if self.eps <= self.eps_min:
            self.eps = self.eps_min
        else:
            self.eps = self.eps * self.eps_decay

    # Function to get the action
    def get_action(self, state: torch.Tensor) -> int:

        # # First update the epsilon
        # self.update_epsilon()

        if random.random() < self.eps:
            return self.env.action_space.sample()
        else:
            output_tensor: torch.Tensor = self.model(state)
            return output_tensor.argmax().item()

    # Function to update the target model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Function to train the model
    # @profile
    def train_model(self):

        # Get the sample from the memory
        sample = self.memory.sample(self.batch_size)

        # Get the states, actions, rewards, next states and dones
        states, actions, rewards, next_states, dones = zip(*sample)

        # Convert the state, action, reward, next_state into Tensors
        states       = torch.tensor(np.asarray(states), dtype=torch.float).to(self.model.device)
        actions      = torch.tensor(actions, dtype=torch.long).to(self.model.device)
        rewards      = torch.tensor(rewards, dtype=torch.float).to(self.model.device)
        next_states  = torch.tensor(np.asarray(next_states), dtype=torch.float).to(self.model.device)

        # Get the predicted Q-Values for next_states
        predicted_Q_values: torch.Tensor = self.model(states)

        # The Target Q-Values
        target_Q_values: torch.Tensor = self.target_model(states)

        # Update the target Q-Values
        for idx in range(len(sample)):

            # Get the next state max q-value
            next_state_max_q_value = torch.max(self.model(next_states[idx]))

            # Update the target Q-Value
            Q_new = rewards[idx] if dones[idx] else rewards[idx] + self.discount_factor * next_state_max_q_value

            # Set this value in the target Q-Values
            target_Q_values[idx][actions[idx]] = Q_new

        ## Summary writter every 100 games
        # if TBOARD:
        #     self.writter.add_scalar('Loss', self.loss_fn(predicted_Q_values, target_Q_values), self.games_completed)
        #     self.writter.add_scalar('Epsilon', self.eps, self.games_completed)

        # Remove the Gradient
        self.optimizer.zero_grad()

        # FInd the loss between the predicted Q-Values and the target Q-Values
        loss: torch.Tensor = self.loss_fn(predicted_Q_values, target_Q_values)

        # Backpropagate the loss
        loss.backward()

        # Update the weights
        self.optimizer.step()

    # Function to train the model
    def training_loop(self, tick_speed: int = 200):

        # Run the training loop for the total number of games
        for _ in tqdm(range(self.total_games), desc='Game: '):

            # Reset the environment
            state, _ = self.env.reset()
            # state = torch.unsqueeze(torch.FloatTensor(state), 0)
            total_move_count = 0

            # Run the episode
            while True:

                # This is placed so that pygames doesn't crash
                if self.env.render_mode == 'human':
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            raise SystemExit

                # Get the action
                action = self.get_action(torch.FloatTensor(state))

                # Take the action
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Store the transition in the memory
                self.memory.push(state=state, action=action, next_state=next_state, reward=reward, done=terminated or truncated)

                # Move to the next state
                state = next_state

                # Update the steps done and increment the move count
                self.steps_done += 1
                total_move_count += 1

                # If the episode is done, break the loop
                if terminated or truncated:
                    break

                # Tick
                if self.env.render_mode == 'human':
                    self.env.renderer.clock.tick(tick_speed)

            # Update the games completed
            self.games_completed += 1

            # Update the target model
            if self.games_completed % self.target_update_freq == 0:
                self.update_target_model()

            # Save the checkpoint
            if self.games_completed % CHECK_POINT_FREQ == 0:
                self.save_checkpoint()

            # Train the model
            self.train_model()

            # Update the epsilon
            self.update_epsilon()

            # Logger
            logger.info(f'Game: {self.games_completed}, Epsilon: {self.eps}, Move Count: {total_move_count}, Score: {self.env.snake.score}')

            # # Summary writter to record score for every game
            # if TBOARD:
            #     self.writter.add_scalar('Score', self.env.snake.score, self.games_completed)
            #     self.writter.add_scalar('Move Count', total_move_count, self.games_completed)

        # Save the model
        try:
            self.model.save_model(os.path.join('models', 'dqn'), MODEL_NAME)
        except Exception as e:
            logger.error(f'Error while saving the model: {e}')
            logger.debug('Model saved with name "model"')
            self.model.save_model(os.path.join('models', 'dqn'), 'model')

    # Saving Check point
    def save_checkpoint(self):

        # First check if the directory exists
        path = os.path.join('checkpoints', 'dqn', MODEL_NAME)
        if not os.path.exists(path):
            os.makedirs(path)

        # Check if the folder has json file
        if not os.path.exists(os.path.join(path, 'record.json')):
            with open(os.path.join(path, 'record.json'), 'w') as f:
                json.dump({}, f)

        # Load the json file
        with open(os.path.join(path, 'record.json'), 'r') as f:
            record = json.load(f)

        # Create name of the checkpoint using the game number
        name = f'checkpoint_{self.games_completed}.pth'

        # Save the model and target model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn,
            'eps': self.eps,
            'steps_done': self.steps_done,
            'games_completed': self.games_completed,
            'memory': self.memory,
        }, os.path.join(path, name))

        # Update the record
        record[name] = {
            'eps': self.eps,
            'steps_done': self.steps_done,
            'games_completed': self.games_completed
        }

        # Save the record
        with open(os.path.join(path, 'record.json'), 'w') as f:
            json.dump(record, f)

        # Logging
        logger.info(f'Checkpoint saved: {name}')

    # Loading Check point
    def load_checkpoint(self, checkpoint_name: str):

        # First check if the directory exists
        path = os.path.join('checkpoints', 'dqn', MODEL_NAME)

        # Load the json file
        with open(os.path.join(path, 'record.json'), 'r') as f:
            record = json.load(f)

        # Check if the checkpoint exists
        if checkpoint_name not in record:
            raise Exception('Checkpoint not found')

        # Load the checkpoint
        checkpoint = torch.load(os.path.join(path, checkpoint_name))

        # Load the model and target model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])

        # Load the optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load the loss function
        self.loss_fn = checkpoint['loss']

        # Load the epsilon
        self.eps = checkpoint['eps']

        # Load the steps done
        self.steps_done = checkpoint['steps_done']

        # Load the games completed
        self.games_completed = checkpoint['games_completed']

        # Load the memory
        self.memory = checkpoint['memory']

        # logger
        logger.info(f'Checkpoint loaded: {checkpoint_name}')

    # Load the trained model
    def load_trained_model(self, model_name: str):

        # Get details from the name
        if model_name != 'model':
            try:
                _, dimensions, hidden_layer_sizes, _, _  = model_name.split('-')
            except ValueError:
                _, dimensions, hidden_layer_sizes, _, _, _, _, _  = model_name.split('-')

            # Create the environment
            self.env.board_height = int(dimensions.split('x')[0])
            self.env.board_width  = int(dimensions.split('x')[1])
            hidden_layer_sizes = [int(x.strip()) for x in hidden_layer_sizes.split(',')]

            # Create the model
            self.model = QNetwork(
                input_size=self.env.observation_space.shape[0],
                output_size=self.env.action_space.n,
                hidden_layer_sizes=hidden_layer_sizes
            )

        # Load the model
        self.model.load_model(os.path.join('models', 'dqn'), model_name)

        logger.debug(f'Loaded the model: {model_name}')

    # Function to play the game
    def play_game(self, tick_speed: int = 200):

        # Reset the environment
        state, _ = self.env.reset()

        # Run the episode
        while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

            # Get the action
            output_tensor: torch.Tensor = self.model(torch.FloatTensor(state))
            action = output_tensor.argmax().item()

            # Take the action
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            # Move to the next state
            state = next_state

            # If the episode is done, break the loop
            if terminated or truncated:
                break

            # Tick
            self.env.renderer.clock.tick(tick_speed)

        # Logger
        logger.info(f'Score: {self.env.snake.score}')
