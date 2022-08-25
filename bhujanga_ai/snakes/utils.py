from collections import deque
from .basesnake import BaseSnake
from helper import Point, Direction
import logging
import os
import configparser
from copy import deepcopy
import numpy as np
from loguru import logger as lg


# Torch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module

# Setup Config File
config = configparser.ConfigParser()
config.read(r'bhujanga_ai\settings.ini')


# Required Constants
COMPLETE_MODEL_DIR = config['GAME - BASIC']['COMPLETE_MODEL_DIR']
CHECKPOINT_DIR = config['GAME - BASIC']['CHECKPOINT_DIR']


def Setup_Logging():
    # Setting up the logger
    logger = logging.getLogger(__name__)
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


# BASE CLASS FOR FINDERS
class Finder:
    def __init__(self, snake: BaseSnake, end: Point, log: bool = False, debug: bool = False):
        self.snake = snake
        self.start = snake.head
        self.end = end
        self.logging = log
        self.debug = debug
        self.visited = []
        self.path = {}

    def find_path(self):
        # Will be overridden by child classes
        pass

    def get_path_directions(self) -> list:
        # sourcery skip: de-morgan, dict-comprehension, merge-else-if-into-elif, simplify-len-comparison, swap-if-else-branches, use-named-expression, while-guard-to-condition

        # Starting from the end, backtrack the path
        current = self.end
        path = [current]
        while current != self.start:
            if current.parent is None:
                break
            current = current.parent
            path.append(current)

        if self.start not in path:
            path.append(self.start)
        if self.debug:
            logger.debug(f"Path from {self.start} to {self.end}: {path[::-1]}")

        # Convert path to list of directions
        directions = {}
        for i in range(len(path) - 1):
            directions[path[i + 1]] = path[i] - path[i + 1]

        # Check if path is empty
        if len(directions) == 0:
            if self.debug:
                logger.debug('Path is empty')
        else:
            if self.debug:
                logger.debug('Got directions')
            self.path = directions

    def copy(self) -> 'Finder':
        # Create a deep copy of the finder
        return deepcopy(self)

    def path_exists(self) -> bool:
        return len(self.path) > 0 if self.path else False


# BREADTH FIRST SEARCH
class BFS_Finder(Finder):

    def get_neighbors(self, current: Point, exclude_tail: bool = False) -> list:

        # Board range
        pos = current.copy()
        x_range = range(self.snake.board_width)
        y_range = range(self.snake.board_height)

        # Get all neighbours
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        if current == self.snake.head:
            neighbors = [pos + direction for direction in directions if direction != -self.snake.direction]
        else:
            neighbors = [pos + direction for direction in directions]

        # Remove invalid neighbors
        allowed_neighbors = deepcopy(neighbors)
        for point in neighbors:

            #  Remove if out of bounds
            if point.x not in x_range or point.y not in y_range or point == self.snake.head:
                allowed_neighbors.remove(point)

            # Remove if it is in snakes body
            elif exclude_tail:
                body = deepcopy(self.snake.body)
                body.pop()
                if point in body:
                    allowed_neighbors.remove(point)

            elif point in self.snake.body:
                allowed_neighbors.remove(point)

        # if self.debug:
        #     logger.debug(f'Neighbor for point {current} : {allowed_neighbors}')

        return allowed_neighbors

    def find_path(self, exclude_tail: bool = False) -> None:

        if self.debug:
            logger.debug('Starting BFS')
            logger.debug(f'Finding path from Start : {self.start} to End : {self.end}')
            logger.debug(f'Excluding Tail : {exclude_tail}')
            # logger.debug(f'Walls are : {list(self.snake.body)[:-1]}')

        # Initialize the queue and visited list
        self.queue = deque()
        self.visited = []
        self.path = {}

        # Mark the start node as visited and enqueue it
        self.queue.append(self.start)

        # if self.debug:
        #     logger.debug(f'Initial queue - {self.queue}')

        while self.queue:

            # Dequeue a vertex from queue
            current: Point = self.queue.popleft()
            # if self.debug:
            #     logger.debug(f'Current point: ({current.x}, {current.y})')

            # Get all adjacent vertices of the dequeued vertex
            if current not in self.visited:
                self.visited.append(current)

                # Get neighbours from grid
                # if self.debug:
                #     logger.debug(f'Getting neighbors for point: {current}')

                for neighbour in self.get_neighbors(current, exclude_tail):
                    neighbour: Point
                    if neighbour not in self.visited:
                        neighbour.parent = current
                        self.queue.append(neighbour)

                        if neighbour == self.end:
                            self.end.parent = current

                            if self.debug:
                                logger.debug('Found path')

                            self.get_path_directions()
                            return

            # if self.debug:
            #     logger.debug(f'While loop running - Total visited - {len(self.visited)}')

        if self.debug:
            logger.debug('No path found')


# Neural Network for Basic Q-Learning Agent
class Q_Network_Basic(Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int) -> None:

        # Initialize the Super Class
        super().__init__()

        # Define the layers
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        # Hidden layers (if any)
        if len(hidden_sizes) > 1:
            self.hidden_layers = [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)]
        else:
            self.hidden_layers = []

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # The input layer pass
        x = F.relu(self.input_layer(x))

        # Hidden layers pass (if any)
        if len(self.hidden_layers) > 0:
            for layer in self.hidden_layers:
                x = F.relu(layer(x))

        # Output layer pass
        x = self.output_layer(x)

        return x

    # Save the model
    def save(self, file_name: str = 'model.pth') -> None:
        """
        Save the model
        """

        if not os.path.exists(COMPLETE_MODEL_DIR):
            os.mkdir(COMPLETE_MODEL_DIR)

        torch.save(self.state_dict(), os.path.join(COMPLETE_MODEL_DIR, file_name))

    # Load the model
    def load(self, file_name: str = 'model.pth') -> None:
        """
        Load the model
        """

        self.load_state_dict(torch.load(os.path.join(COMPLETE_MODEL_DIR, file_name)))

    # Save Checkpoints
    def save_checkPoints(self, state, checkpoint_name) -> None:

        # Save the model
        if not os.path.exists(CHECKPOINT_DIR):
            os.mkdir(CHECKPOINT_DIR)
        fpath = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        torch.save(state, fpath)

    # Load Checkpoints
    def load_checkPoints(self, model, trainer, checkpoint_name) -> None:
        """
        Load the model
        """
        # Get the checkpoint path
        fpath = os.path.join(CHECKPOINT_DIR, checkpoint_name)

        # Check if the checkpoint exists
        if not os.path.exists(fpath):
            raise FileNotFoundError('No model to load')

        # Load the checkpoint
        checkpoint = torch.load(fpath)
        model.load_state_dict(checkpoint['state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])

        return model, trainer, checkpoint['epoch']


# Trainer for Q-Learning Agent
class Q_Trainer_Basic:

    # Initialize the trainer
    def __init__(self, model: Q_Network_Basic, learning_rate: float = 0.001, gamma: float = 0.9) -> None:

        self.lr         = learning_rate
        self.gamma      = gamma
        self.model      = model
        self.optimizer  = optim.Adam(self.model.parameters(), lr=self.lr)  # Adam Optimizer
        self.criterion  = nn.MSELoss()  # Mean Squared Error Loss

    # Train the model
    @lg.catch
    def train_step(self, state, action, reward, next_state, done) -> None:

        # Inputs can be single or batch
        # Hence convert them into tensor
        state       = torch.tensor(np.array(state), dtype=torch.float)
        action      = torch.tensor(action, dtype=torch.long)
        reward      = torch.tensor(reward, dtype=torch.float)
        next_state  = torch.tensor(np.array(next_state), dtype=torch.float)

        # If its just single input
        if len(state.shape) == 1:

            # Input will be of the form (1, x)
            state       = torch.unsqueeze(state, dim=0)
            action      = torch.unsqueeze(action, dim=0)
            reward      = torch.unsqueeze(reward, dim=0)
            next_state  = torch.unsqueeze(next_state, dim=0)
            done        = (done, )

        # 1. Predicted Q values with current state
        # This runs the model on the current state
        # By running it means we call the forward pass function and get the output x as our predicted Q value
        pred = self.model(state)

        # 2. Predicted Q values with next state - r + gamma * max(pred)
        target = pred.clone()

        for idx in range(len(done)):

            q_next_max = torch.max(self.model(next_state[idx]))

            # If the episode is done, then the target is the reward
            # Else the target is the reward + gamma * max(pred)
            Q_new = reward[idx] if done[idx] else reward[idx] + self.gamma * q_next_max

            # print("Next max Q (Tensor):", q_next_max)
            # print("Next max Q (Value):", q_next_max.item())
            # print(f'Current Reward [{idx}]:', reward[idx])
            # print("New Q-Value (Tensor):", Q_new)
            # print("New Q-Value (Value):", Q_new.item())
            # print("Actions (Available):", action[idx])
            # print("Max Action (Tensor):", torch.max(action[idx]))
            # print("Max Action (Value):", torch.max(action[idx]).item())
            # print(f"Target[{idx}] (Tensor) ", target[idx])
            # print(f"Target[{idx}][{torch.max(action[idx]).item()}]:", target[idx][torch.max(action[idx]).item()])

            # Set the target value for the action taken
            target[idx][torch.max(action[idx]).item()] = Q_new
            # print(print(f"Target[{idx}] (Updated Tensor) ", target[idx]))
            # print()

        # 3. Calculate loss
        # Loss is the mean of the squared error between the predicted and target values
        # First flush out the previous gradients (if any) using zero_grad()
        self.optimizer.zero_grad()

        # Calculate the loss and backpropagate
        self.loss = self.criterion(target, pred)
        self.loss.backward()

        # 4. Update the weights using the set optimizer (Adam by default)
        self.optimizer.step()
