import os

# [GAME - BASIC]
WIDTH               = 10
HEIGHT              = 10
PYGAME              = False
LAP_TIME            = 10
MEDIA_DIR           = r"""C:\Users\hrush\OneDrive - iitgn.ac.in\Desktop\Projects\Bhujanga-AI\Media"""
COMPLETE_MODEL_DIR  = r"""C:\Users\hrush\OneDrive - iitgn.ac.in\Desktop\Projects\Bhujanga-AI\Models\Complete_Models"""
CHECKPOINT_DIR      = r"""C:\Users\hrush\OneDrive - iitgn.ac.in\Desktop\Projects\Bhujanga-AI\Models\Checkpoints"""
GIF_PATH            = os.path.join(MEDIA_DIR, 'GIFs')
GRAPH_PATH          = os.path.join(MEDIA_DIR, 'Graphs')


# [LOGGING]
LOGGING         = True
DEBUG           = False
LOGGING_PATH    = r"""C:\Users\hrush\OneDrive - iitgn.ac.in\Desktop\Projects\Bhujanga-AI\Logs.log"""
LOG_DIR         = r"""C:\Users\hrush\OneDrive - iitgn.ac.in\Desktop\Projects\Bhujanga-AI\Logs"""
DEBUG_PATH      = r"""C:\Users\hrush\OneDrive - iitgn.ac.in\Desktop\Projects\Bhujanga-AI\Debug.log"""

# [CURSES]
# The speed of the game (In curses, the timeout is in milliseconds, higher is slower)
TIMEOUT = 10
HEAD_CHR = 'H'
BODY_CHR = '#'
TAIL_CHR = 'T'
FOOD_CHR = 'O'

# [PYGAME]
# Define Constants
FONT = r"""C:\Users\hrush\OneDrive - iitgn.ac.in\Desktop\Projects\Bhujanga-AI\Lora-Regular.ttf"""
BLOCKSIZE   = 20
SPEED       = 200
BORDER      = 3

# Colors
BLACK       = (0, 0, 0)
GREY        = (150, 150, 150)
WHITE       = (255, 255, 255)
RED         = (255, 0, 0)
GREEN       = (0, 255, 0)
GREEN2      = (100, 255, 0)
BLUE        = (0, 0, 255)
BLUE2       = (0, 100, 255)

# [RL SNAKE]
MAX_MEMORY          = 100_000
BATCH_SIZE          = 5000
LEARNING_RATE       = 0.01
DISCOUNT_RATE       = 0.95
EPSILON_MAX         = 1.0
EPSILON_MIN         = 0.01
EPSILON_DECAY       = 0.994
EPOCHS              = 1000
HIDDEN_LAYER_SIZES  = (512, 256, 128)
TARGET_UPDATE_FREQ  = 100
