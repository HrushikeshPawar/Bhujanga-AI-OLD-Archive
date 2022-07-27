"""The main game class for the snake game"""
# /bhujanga_ai/game.py


# Import the required modules
import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP

# Import various helper and agent classes
from snakes.basesnake import BaseSnake
from helper import BodyCollisionError, Direction, WallCollisionError


# Required Constants
TIMEOUT = 100  # The speed of the game (In curses, the timeout is in milliseconds, higher is faster)
HEAD_CHR = 'H'
BODY_CHR = '#'
TAIL_CHR = 'T'
FOOD_CHR = 'O'
B_HEIGHT = 20
B_WIDTH = 20


# Required Dicts
key_to_direction = {
    KEY_LEFT: Direction.LEFT,
    KEY_RIGHT: Direction.RIGHT,
    KEY_UP: Direction.UP,
    KEY_DOWN: Direction.DOWN
}


# The Game Class
class Game:
    """The main class which initializes and plays the game"""

    def __init__(
        self,
        height : int,
        width : int,
        random_init : bool = False,
        agent : BaseSnake = BaseSnake,
        verbose : bool = False,
        board : curses.newwin = None,
    ) -> None:
        """Initialize the game"""

        # Initialize the game's environment (board)
        self.board_width = width
        self.board_height = height

        # Initialize the game's agent
        self.agent = agent(height, width, random_init)

        # Initialize the game's verbose flag
        self.verbose = verbose

        # Initialize the game's initial position (snake's head)
        # Here we can take two approaches:
        # 1. Random Initialization
        # 2. Fixed Initialization
        self.agent._initialize_snake(random_init)

        # Place the food at random location on the board
        # I first thought of placing the food at random location as well as fixed location on the board
        # But then I decided to just place it randomly
        self.agent._place_food()

        # Initialize the board
        self.board = board
        self.timeout = TIMEOUT

    # Rendering the Game Board on terminal using curses
    def render_curses(self) -> None:

        # Rendering the Head
        self.board.addstr(self.agent.head.y, self.agent.head.x, HEAD_CHR)

        # Rendering the Body
        if len(self.agent.body) > 1:
            for i in range(len(self.agent.body) - 1):
                self.board.addstr(self.agent.body[i].y, self.agent.body[i].x, BODY_CHR)

        # Rendering the Tail
        if len(self.agent.body) > 0:
            self.board.addstr(list(self.agent.body)[-1].y, list(self.agent.body)[-1].x, TAIL_CHR)
        # if self.agent.tail:
        #     self.board.addstr(self.agent.tail.y, self.agent.tail.x, TAIL_CHR)

        # Rendering the Food
        self.board.addstr(self.agent.food.y, self.agent.food.x, FOOD_CHR)

        # Render the score counter (for me it is the lenght counter)
        self.board.addstr(0, 5, "Length: " + str(self.agent.score))

        # Refresh the board
        self.board.refresh()


# The Main Initialization Function
def main():

    # Initialize the curses library
    curses.initscr()

    # Disable the echoing of keys to the screen
    curses.noecho()

    # Make the cursor invisible
    curses.curs_set(0)

    # Create the game board window
    board = curses.newwin(B_HEIGHT, B_WIDTH, 0, 0)

    # Set the game speed
    board.timeout(TIMEOUT)

    # Activate keypad mode
    board.keypad(1)

    # Draw the game board border
    board.border(0)

    # Initialize the Game
    game = Game(B_HEIGHT, B_WIDTH, random_init=False, agent=BaseSnake, verbose=False, board=board)

    # The main game loop
    while True:

        # Clear the screen and render the game board
        board.clear()
        board.border(0)

        # Render the game board
        game.render_curses()

        # Get the key pressed by the user
        key = board.getch()

        # Check if the user pressed the exit key
        if key == ord('q'):
            break

        # Check if the user pressed the arrow key
        # If yes then update the direction of the snake
        # Check for collisions
        try:
            if key in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN]:
                game.agent.move(key_to_direction[key])

            # If no direction key is pressed, continue moving in the same direction
            elif key == -1:
                # print(game.agent)
                game.agent.move(game.agent.direction)

        except WallCollisionError:
            print("Wall Collision Error")
            break

        except BodyCollisionError:
            print("Body Collision Error")
            break

    # End the curses session
    curses.endwin()
    print("Game Over")
    print('Your score is: ' + str(game.agent.score))


# Run the main function
if __name__ == "__main__":
    main()
