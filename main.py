import time

import gymnasium
from loguru import logger as lg

# Importing the bhujanga_gym package
import bhujanga_gym.settings as settings
# import DQN Agent
from bhujanga_ai.dqn import DQNAgent, writter


# Planned play
def planned_play():

    import pygame

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode="human")

    # Creating the agent
    first_moves = [2, 2, 1, 1]

    # Reset the environment
    env.reset()

    # Play first moves
    for action in first_moves:
        _, _, done, _, _ = env.step(action)
        env.renderer.clock.tick(10)

    loop_moves = [0] + [3] * 4 + [0] + [1] * 4 + [0] + [3] * 4 + [0] + [1] * 4 + [0] + [3] * 5 + [2] * 5 + [1] * 5

    while not done:
        for action in loop_moves:
            _, reward, done, trunc, _ = env.step(action)
            env.renderer.clock.tick(10)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

            done = done or trunc

            print(f'Reward: {reward}')

            if done:
                break

    env.close()


# Human Play
def human_play():

    import pygame

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode="human")

    obs = env.reset()
    action = 1

    while True:

        # Check for events
        for event in pygame.event.get():

            # Check key press
            if event.type == pygame.KEYDOWN:

                # Check for up key
                if event.key == pygame.K_UP:
                    action = 0

                # Check for right key
                if event.key == pygame.K_RIGHT:
                    action = 1

                # Check for down key
                if event.key == pygame.K_DOWN:
                    action = 2

                # Check for left key
                if event.key == pygame.K_LEFT:
                    action = 3

                # Check for quit key
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()

        # Move the snake
        obs, reward, done, trun, info = env.step(action)

        if done or trun:
            env.reset()

        env.renderer.clock.tick(3)


# The main function
@lg.catch
def main():

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode="human")

    # Creating the agent
    agent = DQNAgent(env)

    # Training the agent
    try:
        agent.training_loop()
    except KeyboardInterrupt:
        writter.close()
        lg.info("Training interrupted by user")
        env.close()


# Playing the Game
def play():
    import pygame

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode="human")

    # Creating the agent
    agent = DQNAgent(env)

    agent.load_trained_model('[20221116 235325]-6x6-64,64-2000')

    # Playing the game
    print()
    for _ in range(10):
        agent.play_game(10)
        time.sleep(1)


# Calling the main function
if __name__ == "__main__":
    # main()
    play()
    # human_play()
    # planned_play()
