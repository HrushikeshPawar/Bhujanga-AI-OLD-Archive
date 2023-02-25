from concurrent.futures import ProcessPoolExecutor
import gymnasium
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import FlattenObservation
from loguru import logger as lg
from typing import Optional, List, Union, Tuple

# Importing the bhujanga_gym package
import bhujanga_gym.settings as settings
from bhujanga_gym.envs.snake_world import SnakeWorldEnv
from bhujanga_gym.wrappers.full_view_wrapper import FullViewWrapper, FullViewWithStackWrapper
from bhujanga_gym.wrappers.path_to_tail_wrapper import PathToTailWrapper, PathToTailStackedWrapper
from bhujanga_gym.wrappers.updated_reward import UpdatedRewardWrapper
from bhujanga_gym.wrappers.basic_stacked import BasicStackedWrapper
from bhujanga_gym.settings import BOARD_WIDTH, BOARD_HEIGHT

# import DQN Agent
from bhujanga_ai.dqn import DQNAgent, PERDQNAgent

# Dict of all available agents
agents = {
    "dqn": DQNAgent,
    "per_dqn": PERDQNAgent
}


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
def basic(render_mode: Optional[str] = None, agent_type : str = 'dqn') -> Tuple[SnakeWorldEnv, Union[DQNAgent, PERDQNAgent]]:

    print('The Basic Game')

    # Creating the environment
    # env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode="human")
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=render_mode)

    # Creating the agent
    agent = agents[agent_type](env, addon=f'basic_{agent_type}')

    return env, agent


# Updated reward wrapper game
@lg.catch
def updated_reward_wrapper_game(render_mode: Optional[str] = None, agent_type : str = 'dqn') -> Tuple[SnakeWorldEnv, Union[DQNAgent, PERDQNAgent]]:

    print('Updated Reward Game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=render_mode)

    # Applying the wrapper
    env = UpdatedRewardWrapper(env)

    # Creating the agent
    agent = agents[agent_type](env, addon=f'updated_reward_{agent_type}')

    return env, agent


# Full View wrapper game
@lg.catch
def full_view_wrapper_game(render_mode: Optional[str] = None, agent_type : str = 'dqn') -> Tuple[SnakeWorldEnv, Union[DQNAgent, PERDQNAgent]]:
    print('Full View Game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=render_mode)

    # Applying the wrapper
    env = FullViewWrapper(env)
    print(env.observation_space)

    # Flatten the observation
    env = FlattenObservation(env)

    print(env.observation_space)

    # Creating the agent
    agent = agents[agent_type](env, addon=f'fullview_{agent_type}')

    return env, agent


# Full View Stack with Updated Reward
@lg.catch
def full_view_with_updated_reward_wrapper_game(render_mode: Optional[str] = None, agent_type : str = 'dqn') -> Tuple[SnakeWorldEnv, Union[DQNAgent, PERDQNAgent]]:
    print('Full View with updated reward wrapper game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=render_mode)

    # Applying the wrapper
    env = FullViewWrapper(env)
    print(env.observation_space)

    # Flatten the observation
    env = FlattenObservation(env)
    print(env.observation_space)

    # Reward Wrapper
    env = UpdatedRewardWrapper(env)

    # Creating the agent
    agent = agents[agent_type](env, addon=f'fullview_updated_reward_{agent_type}')

    return env, agent


# Full View with stack wrapper
@lg.catch
def full_view_stack_wrapper_game(render_mode: Optional[str] = None, agent_type : str = 'dqn') -> Tuple[SnakeWorldEnv, Union[DQNAgent, PERDQNAgent]]:
    print('Full View Stack Game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=render_mode)

    # Applying the wrapper
    env = FullViewWithStackWrapper(env)
    print(env.observation_space)

    # Flatten the observation
    env = FlattenObservation(env)

    print(env.observation_space)

    # Creating the agent
    agent = agents[agent_type](env, addon=f'fullview_stack_{agent_type}')

    return env, agent


# Full View Stack with Updated Reward
@lg.catch
def full_view_stack_with_updated_reward_wrapper_game(render_mode: Optional[str] = None, agent_type : str = 'dqn') -> Tuple[SnakeWorldEnv, Union[DQNAgent, PERDQNAgent]]:
    print('Full View stack with updated reward wrapper game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=render_mode)

    # Applying the wrapper
    env = FullViewWithStackWrapper(env)
    print(env.observation_space)

    # Flatten the observation
    env = FlattenObservation(env)
    print(env.observation_space)

    # Reward Wrapper
    env = UpdatedRewardWrapper(env)

    # Creating the agent
    agent = agents[agent_type](env, addon=f'fullview_stack_updated_reward_{agent_type}')

    return env, agent


# Path to tail wrapper game
@lg.catch
def path_to_tail_wrapper_game(render_mode: Optional[str] = None, agent_type : str = 'dqn') -> Tuple[SnakeWorldEnv, Union[DQNAgent, PERDQNAgent]]:
    print('Path to Tail Game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=render_mode)

    # Applying the wrapper
    env = PathToTailWrapper(env)
    print(env.observation_space)

    # Creating the agent
    agent = agents[agent_type](env, addon=f'path_to_tail_{agent_type}')

    return env, agent


# Path to tail with reward wrapper game
@lg.catch
def path_to_tail_with_reward_wrapper_game(render_mode: Optional[str] = None, agent_type : str = 'dqn') -> Tuple[SnakeWorldEnv, Union[DQNAgent, PERDQNAgent]]:
    print('Path to tail with reward wrapper game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=render_mode)

    # Applying the observation wrapper
    env = PathToTailWrapper(env)
    print(env.observation_space)

    # Applying the reward wrapper
    env = UpdatedRewardWrapper(env)

    # Creating the agent
    agent = agents[agent_type](env, addon=f'path_to_tail_with_reward_{agent_type}')

    return env, agent


# Basic Stacked Game
@lg.catch
def basic_stacked_game(render_mode: Optional[str] = None, agent_type : str = 'dqn') -> Tuple[SnakeWorldEnv, Union[DQNAgent, PERDQNAgent]]:
    print('Basic Stacked Game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=render_mode)

    # Applying the Basic Stacked wrapper
    env = BasicStackedWrapper(env, stack_size=4)
    print(env.observation_space)

    # Apply the flatten observation wrapper
    env = FlattenObservation(env)
    print(env.observation_space)

    # Creating the agent
    agent = agents[agent_type](env, addon=f'basic_stacked_{agent_type}')

    return env, agent


# Basic Stacked with updated reward game
@lg.catch
def basic_stacked_with_updated_reward_game(render_mode: Optional[str] = None, agent_type : str = 'dqn') -> Tuple[SnakeWorldEnv, Union[DQNAgent, PERDQNAgent]]:
    print('Basic Stacked with updated reward game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=render_mode)

    # Applying the Basic Stacked wrapper
    env = BasicStackedWrapper(env, stack_size=4)
    print(env.observation_space)

    # Apply the flatten observation wrapper
    env = FlattenObservation(env)
    print(env.observation_space)

    # Applying the reward wrapper
    env = UpdatedRewardWrapper(env)

    # Creating the agent
    agent = agents[agent_type](env, addon=f'basic_stacked_updated_reward_{agent_type}')

    return env, agent


# Path to tail stacked with updated reward game
@lg.catch
def path_to_tail_stacked_with_updated_reward_game(render_mode: Optional[str] = None, agent_type : str = 'dqn') -> Tuple[SnakeWorldEnv, Union[DQNAgent, PERDQNAgent]]:
    print('Path to tail stacked with updated reward game')

    # Creating the environment
    env = gymnasium.make("bhujanga_gym/SnakeWorld-v0", render_mode=render_mode)

    # Applying the Basic Stacked wrapper
    env = PathToTailStackedWrapper(env, stack_size=4)
    print(env.observation_space)

    # Apply the flatten observation wrapper
    env = FlattenObservation(env)
    print(env.observation_space)

    # Applying the reward wrapper
    env = UpdatedRewardWrapper(env)

    # Creating the agent
    agent = agents[agent_type](env, addon=f'path_to_tail_stacked_updated_reward_{agent_type}')

    return env, agent


# Create Dict of all avaiilable models
models = {
    "basic": {
        'function': basic,
        'model_file_name': '[20221121 005116]-6x6-512,512-128-0.01-0.95-5000-basic'
    },
    "updated_reward": {
        'function': updated_reward_wrapper_game,
        'model_file_name': '[20221121 125201]-6x6-512,512-128-0.01-0.95-10000-updated_reward'
    },
    "fullview": {
        'function': full_view_wrapper_game,
        'model_file_name': '[20221121 013542]-6x6-512,512-128-0.01-0.95-5000-fullview'
    },
    "fullview_updated_reward": {
        'function': full_view_with_updated_reward_wrapper_game,
        'model_file_name': '[20221121 015123]-6x6-512,512-128-0.01-0.95-5000-fullview_updated_reward'
    },
    "fullview_stack": {
        'function': full_view_stack_wrapper_game,
        'model_file_name': '[20221121 020443]-6x6-512,512-128-0.01-0.95-5000-fullview_stack'
    },
    "fullview_stack_updated_reward": {
        'function': full_view_stack_with_updated_reward_wrapper_game,
        'model_file_name': '[20221121 094552]-6x6-512,512-128-0.01-0.95-5000-fullview_stack_updated_reward'
    },
    "path_to_tail": {
        'function': path_to_tail_wrapper_game,
        'model_file_name': '[20221121 021724]-6x6-512,512-128-0.01-0.95-5000-path_to_tail'
    },
    "path_to_tail_updated_reward": {
        'function': path_to_tail_with_reward_wrapper_game,
        'model_file_name': '[20221121 023532]-6x6-512,512-128-0.01-0.95-5000-path_to_tail_with_reward'
    },
    "basic_stacked": {
        'function': basic_stacked_game,
        'model_file_name': '[20221121 104618]-6x6-512,512-128-0.01-0.95-10000-basic_stacked'
    },
    "basic_stacked_updated_reward": {
        'function': basic_stacked_with_updated_reward_game,
        'model_file_name': '[20221121 140733]-6x6-512,512-128-0.01-0.95-10000-basic_stacked_updated_reward'
    },
    "path_to_tail_stacked_updated_reward": {
        'function': path_to_tail_stacked_with_updated_reward_game,
        'model_file_name': '[20221121 151420]-6x6-512,512-128-0.01-0.95-10000-path_to_tail_stacked_updated_reward'
    }
}


# Main function
@lg.catch
def train_models(selected_models: Optional[List[str]] = None, render_mode: Optional[str] = None, selected_agents: Optional[List[str]] = None) -> None:

    render_mode = render_mode

    # Setting the types
    agent: DQNAgent
    env: SnakeWorldEnv

    if selected_models is None:
        selected_models = list(models.keys())

    if selected_agents is None:
        selected_agents = ['dqn']

    # Create a dict to store the results
    for model in selected_models:

        # Print the model name
        print(f"\n\nModel: {model}")

        for agent_type in selected_agents:
            print(f"Agent: {agent_type}")

            # Create the environment and agent
            env, agent = models[model]['function'](render_mode=render_mode, agent_type=agent_type)

            # Training the agent
            try:
                agent.training_loop()
            except KeyboardInterrupt:
                lg.info("Training interrupted by user, saving a check point")
                agent.save_checkpoint()

            # Close the environment
            env.close()


# Playing the Game
@lg.catch
def play(model: str, model_file_name: Optional[str] = None):
    # Setting the types
    agent: DQNAgent
    env: SnakeWorldEnv

    function = models[model]['function']
    if model_file_name is None:
        model_file_name = models[model]['model_file_name']

    # Create the environment and the agent
    env, agent = function(render_mode="human")

    # Load the model
    agent.load_trained_model(model_file_name)

    # Max Score possible
    max_score = env.total_points_to_earn

    # Play the game for 10 episodes
    scores = []
    for i in range(10):

        # Play the game
        agent.play_game(15)

        # Store the score
        scores.append(agent.env.snake.score)
        print(f'Game {i + 1} Score: {agent.env.snake.score}')

    # Calculate the average score
    avg_score = sum(scores) / len(scores)
    print(f'Average Score: {avg_score}')
    print(f'Average Percentage Score: {avg_score / max_score * 100}\n\n')

    # Close the environment
    env.close()


# Compare the models
@lg.catch
def compare_models():
    # Setting the types
    agent: DQNAgent
    env: SnakeWorldEnv

    # Create a dict to store the results
    results = {}
    for model in selected_models:

        # Print the model name
        print(f"\n\nModel: {model}")

        results[model] = {
            'scores': [],
            'average_score': 0,
        }

        # Create the environment and agent
        env, agent = models[model]['function'](render_mode='human')

        # Load the model
        agent.load_trained_model(models[model]['model_file_name'])

        # Play for 10 games and store the results
        print(f'Max Score possible: {BOARD_HEIGHT * BOARD_WIDTH - 2}')
        for i in range(10):

            # Play the game
            agent.play_game(50)

            # Store the score
            results[model]['scores'].append(agent.env.snake.score)
            print(f'Game {i + 1} Score: {agent.env.snake.score}')

        # Calculate the average score
        results[model]['average_score'] = sum(results[model]['scores']) / len(results[model]['scores'])
        print(f'Average Score: {results[model]["average_score"]}')

        # Close the environment
        env.close()


# Train the given model name
@lg.catch
def train(model: str, render_mode: Optional[str] = None):
    # Setting the types
    agent: DQNAgent
    env: SnakeWorldEnv

    # Create the environment and agent
    env, agent = models[model]['function'](render_mode=render_mode)

    # Training the agent
    try:
        agent.training_loop()
    except KeyboardInterrupt:
        lg.info("Training interrupted by user, saving a check point")
        agent.save_checkpoint()
        env.close()

    # Close the environment
    env.close()


# Train 4 agents with ProcessPoolExecutor
@lg.catch
def train_with_executor():
    with ProcessPoolExecutor() as executor:
        executor.map(train, selected_models)


# Vectorized training
@lg.catch
def vectorized_env():

    # Create the environment
    env = SyncVectorEnv(
        lambda: FlattenObservation(BasicStackedWrapper(gymnasium.make('bhujanga_gym/SnakeWorld-v0'))) for _ in range(4)
    )

    # The observation and action spaces
    print(f"\nObservation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    # The single observation and action spaces
    print(f"\nSingle Observation Space: {env.single_observation_space}")
    print(f"Single Action Space: {env.single_action_space}")


# Calling the main function
if __name__ == "__main__":
    selected_models = [
        # 'basic',
        # 'updated_reward',
        'fullview',
        # 'fullview_updated_reward',
        # 'fullview_stack',
        # 'fullview_stack_updated_reward',
        # 'path_to_tail',
        # 'path_to_tail_updated_reward',
        # 'basic_stacked',
        # 'basic_stacked_updated_reward',
        # 'path_to_tail_stacked_updated_reward',
    ]

    selected_agents = [
        'dqn',
        'per_dqn'
    ]
    train_models(selected_models=selected_models, selected_agents=selected_agents)
    # play("path_to_tail_stacked_updated_reward")
    # human_play()
    # planned_play()
    # compare_models()
    # train_with_executor()
    # play('updated_reward', '[20221122 133023]-6x6-512,512,512-128-0.01-0.95-20000-updated_reward')
    # play('path_to_tail_updated_reward', '[20221122 133023]-6x6-512,512,512-128-0.01-0.95-20000-path_to_tail_with_reward')
    # play('basic_stacked_updated_reward', '[20221122 133023]-6x6-512,512,512-128-0.01-0.95-20000-basic_stacked_updated_reward')
    # play('path_to_tail_stacked_updated_reward', '[20221122 133023]-6x6-512,512,512-128-0.01-0.95-20000-path_to_tail_stacked_updated_reward')
    # vectorized_env()
