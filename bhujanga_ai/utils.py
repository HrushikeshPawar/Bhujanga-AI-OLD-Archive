from collections import deque, namedtuple
import logging
import os
import configparser
from loguru import logger as lg
import random
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional


# Torch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

# Setup Config File
config = configparser.ConfigParser()
config.read(r'bhujanga_ai\settings.ini')


# QNetwork Class
class QNetwork(Module):

    # Constructor
    def __init__(self, input_size, output_size, hidden_layer_sizes):
        super(QNetwork, self).__init__()

        # Set the layer sizes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes

        # Create the layers
        # The Input Layer
        self.input_layers = nn.Linear(self.input_size, self.hidden_layer_sizes[0])

        # The Hidden Layers
        if len(self.hidden_layer_sizes) > 1:
            hidden_layer_sizes = zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])
            self.hidden_layers = nn.ModuleList([nn.Linear(h1, h2) for h1, h2 in hidden_layer_sizes])

        # The Output Layer
        self.output = nn.Linear(self.hidden_layer_sizes[-1], self.output_size)

        # Select the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the model to the device
        self.to(self.device)

    # Forward Pass
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # The Input Layer
        x = F.relu(self.input_layers(state))

        # The Hidden Layers
        if len(self.hidden_layer_sizes) > 1:
            for hidden_layer in self.hidden_layers:
                x = F.relu(hidden_layer(x))

        # The Output Layer
        x = self.output(x)

        return x

    # Save the model
    def save_model(self, model_dir: str, model_name: str):
        # Check if the directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the model
        torch.save(self.state_dict(), os.path.join(model_dir, model_name))

    # Load the model
    def load_model(self, model_dir: str, model_name: str):
        # Load the model
        self.load_state_dict(torch.load(os.path.join(model_dir, model_name)))


# Setting up Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# The Replay Memory Class
class ReplayMemory:

    # Constructor
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    # Pushing to the memory
    def push(self, state, action, reward, next_state, done) -> None:
        self.memory.append(Transition(state=state, action=action, next_state=next_state, reward=reward, done=done))

    # Sampling from the memory
    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return random.sample(self.memory, len(self.memory))

        return random.sample(self.memory, batch_size)

    # Getting the length of the memory
    def __len__(self):
        return len(self.memory)


# Prioritized Replay Memory
class PrioritizedReplayMemory:

    # Constructor
    def __init__(self, capacity: int):
        self.capacity = capacity

        # To store the experiences
        self.memory = deque(maxlen=capacity)

        # To store the priorities of the experiences
        self.priorities = deque(maxlen=capacity)

    # Pushing to the memory
    def push(self, state: Tuple, action: int, reward: float, next_state: Tuple, done: bool):

        # First store the experiences in memory
        self.memory.append(Transition(state=state, action=action, next_state=next_state, reward=reward, done=done))

        # As this is a new experience, we assign it the maximum priority
        self.priorities.append(max(self.priorities, default=1.0))

    # Get the probabilities of the priorities attached to the experiences
    def get_probabilities(self, alpha: float = 0.6) -> NDArray:
        # Get the priorities as numpy array
        priorities = np.asarray(self.priorities)

        # Scale the priorities by the alpha value
        scaled_priorities = priorities ** alpha

        # Normalize the scaled priorities to get the probabilities
        probabilities = scaled_priorities / np.sum(scaled_priorities)

        return probabilities

    # Get the importance of the selected probabilities
    def get_importance(self, probabilities: NDArray, beta: float) -> NDArray:
        # Get the importance of the selected probabilities
        importance = np.power(len(self.memory) * probabilities, -beta)

        # Normalize the importance
        importance_normalized = importance / np.max(importance)

        return importance_normalized

    # Sampling from the memory
    def sample(self, batch_size: int, alpha: float = 0.6, beta: float = 0.01) -> Tuple[NDArray, NDArray, NDArray]:

        # Get the sample size
        sample_size = max(len(self.memory), batch_size)

        # Get the probabilities of the priorities
        sample_probabilities = self.get_probabilities(alpha)

        # Select the indices of the experiences based on the probabilities
        sample_indices = np.random.choice(len(self.memory), size=sample_size, p=sample_probabilities)

        # Get the samples from indices
        samples: NDArray = np.asarray(self.memory, dtype=object)[sample_indices]

        # Get the importance of the selected samples
        sample_importance = self.get_importance(sample_probabilities[sample_indices], beta)

        # Return the samples, their indices and the importance
        return samples, sample_indices, sample_importance

    # Update the priorities of the experiences
    def update_priorities(self, indices: NDArray, errors: NDArray, offset: Optional[float] = 0.1) -> None:

        # Update the priorities
        for index, error in zip(indices, errors):
            self.priorities[index] = np.max(np.abs(error)) + offset

    # Getting the length of the memory
    def __len__(self):
        return len(self.memory)
