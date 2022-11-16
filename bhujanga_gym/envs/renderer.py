# Path: bhujanga_gym\renderer.py

from typing import Optional, Tuple
import contextlib
import pygame

# Settting from bhujanga_gym\settings.py
import bhujanga_gym.settings as settings


# The Renderer class
class SnakeGameRenderer:

    # The Initialization function
    def __init__(
        self,
        width: int = settings.BOARD_WIDTH,
        height: int = settings.BOARD_HEIGHT,
        block_size: int = settings.BLOCKSIZE
    ):
        self.board_width = width
        self.board_height = height
        self.block_size = block_size

        # Setup the pygame window
        # Set the game display
        pygame.init()
        self.display = pygame.display.set_mode(size=(self.board_width * block_size, self.board_height * block_size))

        # Set the game caption
        pygame.display.set_caption('Bhujanga-AI - Snake Game Solver')

        # Game clock
        self.clock = pygame.time.Clock()

        # Set the font
        self.font = pygame.font.Font(settings.FONT, 20)
