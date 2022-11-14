
import gymnasium
import pygame
from loguru import logger as lg

# Importing the bhujanga_gym package
import bhujanga_gym
import bhujanga_gym.settings as settings


# The main function
@lg.catch
def main():

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode="human")
    env.reset()
    action = 1

    while True:

        # Human Play Mode
        env.render()

        # Get the action from the user using pygame events
        if pygame.event.get(pygame.KEYDOWN):
            key_pressed = pygame.key.get_pressed()
            if key_pressed[pygame.K_UP]:
                action = 0
            elif key_pressed[pygame.K_RIGHT]:
                action = 1
            elif key_pressed[pygame.K_DOWN]:
                action = 2
            elif key_pressed[pygame.K_LEFT]:
                action = 3

        obs, reward, terminate, truncate, _ = env.step(action)
        log = f"Action: {action}, Reward: {reward}, Terminate: {terminate}, Truncate: {truncate}"
        print(log)

        if terminate:
            env.reset()
            break

        # Clock tick
        env.renderer.clock.tick(settings.SPEED)


# Calling the main function
if __name__ == "__main__":
    main()
