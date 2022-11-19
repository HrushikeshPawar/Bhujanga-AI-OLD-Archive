import time

import gymnasium
from gymnasium.wrappers import FlattenObservation
from loguru import logger as lg

# Importing the bhujanga_gym package
import bhujanga_gym.settings as settings
from bhujanga_gym.envs.snake_world import SnakeWorldEnv
from bhujanga_gym.wrappers.full_view_wrapper import FullViewWrapper, FullViewWithStackWrapper
from bhujanga_gym.wrappers.path_to_tail_wrapper import PathToTailWrapper
from bhujanga_gym.wrappers.updated_reward import UpdatedRewardWrapper

# import DQN Agent
from bhujanga_ai.dqn import DQNAgent


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
    # env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode="human")
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=None)

    # Creating the agent
    agent = DQNAgent(env)

    # Training the agent
    try:
        agent.training_loop()
    except KeyboardInterrupt:
        lg.info("Training interrupted by user")
        env.close()


# Playing the Game
@lg.catch
def play():
    import pygame

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode="human")

    # # Applying the wrapper
    # env = FullViewWrapper(env)

    # # Apply FUll View stack wrapper
    # env = FullViewWithStackWrapper(env)

    # # Flatten the observation
    # env = FlattenObservation(env)

    # Creating the agent
    agent = DQNAgent(env)

    # agent.load_trained_model('[20221116 235325]-6x6-64,64-2000')
    agent.load_trained_model('[20221118 235421]-6x6-512,512-2000-basic')

    # Playing the game
    print()
    for _ in range(10):
        agent.play_game(10)
        time.sleep(1)


# Full View wrapper game
@lg.catch
def full_view_wrapper_game():
    print('Full View wrapper game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode="human")

    # Applying the wrapper
    env = FullViewWrapper(env)
    print(env.observation_space)

    # Flatten the observation
    env = FlattenObservation(env)

    print(env.observation_space)

    # Creating the agent
    agent = DQNAgent(env)

    # Training the agent
    try:
        agent.training_loop()
    except KeyboardInterrupt:
        lg.info("Training interrupted by user")
        env.close()


# Updated reward wrapper game
@lg.catch
def updated_reward_wrapper_game():

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode="human")

    # Applying the wrapper
    env = UpdatedRewardWrapper(env)

    # Creating the agent
    agent = DQNAgent(env)

    # Training the agent
    try:
        agent.training_loop()
    except KeyboardInterrupt:
        lg.info("Training interrupted by user")
        env.close()


# Full View with stack wrapper
@lg.catch
def full_view_stack_wrapper_game():
    print('Full View stack wrapper game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=None)

    # Applying the wrapper
    env = FullViewWithStackWrapper(env)
    print(env.observation_space)

    # Flatten the observation
    env = FlattenObservation(env)

    print(env.observation_space)

    # Creating the agent
    agent = DQNAgent(env)

    # Training the agent
    try:
        agent.training_loop()
    except KeyboardInterrupt:
        lg.info("Training interrupted by user, saving a check point")
        agent.save_checkpoint()
        env.close()


# Path to tail wrapper game
@lg.catch
def path_to_tail_wrapper_game():
    print('Path to tail wrapper game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode="human")

    # Applying the wrapper
    env = PathToTailWrapper(env)
    print(env.observation_space)

    # Creating the agent
    agent = DQNAgent(env, addon='path_to_tail')

    # Training the agent
    try:
        agent.training_loop()
    except KeyboardInterrupt:
        lg.info("Training interrupted by user")
        env.close()


# Path to tail with reward wrapper game
@lg.catch
def path_to_tail_with_reward_wrapper_game():
    print('Path to tail with reward wrapper game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode="human")

    # Applying the observation wrapper
    env = PathToTailWrapper(env)
    print(env.observation_space)

    # Applying the reward wrapper
    env = UpdatedRewardWrapper(env)

    # Creating the agent
    agent = DQNAgent(env, addon='path_to_tail_with_reward')

    # Training the agent
    try:
        agent.training_loop()
    except KeyboardInterrupt:
        lg.info("Training interrupted by user")
        env.close()


# Full View Stack with Updated Reward
@lg.catch
def full_view_stack_with_updated_reward_wrapper_game():
    print('Full View stack with updated reward wrapper game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=None)

    # Applying the wrapper
    env = FullViewWithStackWrapper(env)
    print(env.observation_space)

    # Flatten the observation
    env = FlattenObservation(env)
    print(env.observation_space)

    # Reward Wrapper
    env = UpdatedRewardWrapper(env)

    # Creating the agent
    agent = DQNAgent(env, addon='fullview_stack_updated_reward')

    # Training the agent
    try:
        agent.training_loop()
    except KeyboardInterrupt:
        lg.info("Training interrupted by user, saving a check point")
        agent.save_checkpoint()
        env.close()


# Calling the main function
if __name__ == "__main__":
    # main()
    # play()
    # human_play()
    # planned_play()
    # full_view_wrapper_game()
    # updated_reward_wrapper_game()
    # full_view_stack_wrapper_game()
    # path_to_tail_wrapper_game()
    # path_to_tail_with_reward_wrapper_game()
    full_view_stack_with_updated_reward_wrapper_game()
