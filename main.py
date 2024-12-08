import argparse
import importlib # to import dinamically the Atari Games
import torch
from agent import Agent
from genetic_algorithm import genetic_algorithm_train, play_game, RandomPolicy, load_hof
from evolutionary_strategy import evolution_strategy_train
import os
from supersuit import frame_stack_v1, resize_v1, frame_skip_v0, agent_indicator_v0



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a DNN agent using GA or ES")
    
    # General arguments
    parser.add_argument("--algorithm", type=str, choices=["GA", "ES"], default="GA", 
                        help="Choose the algorithm to use: GA (Genetic Algorithm) or ES (Evolution Strategies)")
    parser.add_argument("--generations", type=int, default=100, 
                        help="Number of generations for the evolutionary process")
    parser.add_argument("--population", type=int, default=10, 
                        help="Population size for the evolutionary algorithm")
    parser.add_argument("--hof_size", type=int, default=20, 
                        help="Size of the Hall of Fame")
    parser.add_argument("--atari_game", type=str, choices=["boxing_v2", "pong_v3"], default="pong_v3",
                        help="Choose the Atari games to test with")
    parser.add_argument("--initial_mutation_power", type=float, default=0.05, 
                        help="Initial value for the mutation power, i.e. noise standard deviation for weight mutation")
    parser.add_argument("--learning_rate", type=float, default=0.1, 
                        help="Learning rate")
    parser.add_argument("--max_timesteps_per_episode", type=int, default=None, 
                        help="Total timesteps per episode")
    parser.add_argument("--max_evaluation_steps", type=int, default=None, 
                        help="Maximum steps for evaluation")
    parser.add_argument("--input_channels", type=int, default=6, 
                        help="Number of input channels for the observation")
    parser.add_argument("--elites_number", type=int, default=2, 
                        help="Number of elites for each generation")          
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode for detailed logging")
    parser.add_argument("--train", action="store_true",
                        help="Train the models")
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering during training")
    parser.add_argument("--test", action="store_true",
                        help="Test the models")
    parser.add_argument("--env_mode", type=str, choices=["AEC", "parallel"], default="AEC", 
                        help="Choose the environment mode: AEC (Agent-Environment Cycle) or parallel")
    parser.add_argument("--precision", type=str, choices=["float32", "float16"], default="float32",
                        help="Specify the precision for computations: float32 or float16")
    parser.add_argument("--save", action="store_true",
                        help="Save the models")
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable adaptive mutation power, decreasing over time")
    parser.add_argument("--ES_model_to_test", type=str, default=None,
                        help="If testing ES, choose the model to test from CLI")
    parser.add_argument("--GA_hof_to_test", type=str, default=None,
                        help="If testing GA, choose the model to test from CLI")
    parser.add_argument("--play_against_yourself", action="store_true",
                        help="If true, the model plays against itself. If false, it plays against a dummy (RandomPolicy)")
    
    return parser.parse_args()


# Define hyperparameters
class Args:
    def __init__(self, args):

        # Hyperparameters
        self.algorithm = args.algorithm
        self.generations = args.generations
        self.population = args.population
        self.hof_size = args.hof_size
        self.atari_game = args.atari_game
        self.mutation_power = args.initial_mutation_power
        self.learning_rate = args.learning_rate
        self.max_timesteps_per_episode = args.max_timesteps_per_episode
        self.max_evaluation_steps = args.max_evaluation_steps
        self.elites_number = args.elites_number
        self.adaptive = args.adaptive

        # clearly, these are not hyperparam, but it's easy to have everything inside an object
        self.debug = args.debug
        self.train = args.train
        self.test = args.test
        self.input_channels = args.input_channels
        self.render = args.render
        self.env_mode = args.env_mode
        self.precision = args.precision
        self.save = args.save
        self.ES_model_to_test = args.ES_model_to_test
        self.GA_hof_to_test = args.GA_hof_to_test
        self.play_against_yourself = args.play_against_yourself


    def print_attributes(self, args): 
        # Print all attributes except `input_channel`
        attributes = vars(self)  # Get all attributes as a dictionary
        for attr, value in attributes.items():
            if attr != "input_channels":  # Skip `input_channel`
                if attr != "elites_number" or args.algorithm=="GA":
                    if attr != "learning_rate" or args.algorithm=="ES":
                        if attr != "ES_model_to_test" or (args.algorithm == "ES" and args.test and not args.train):
                            if attr != "GA_hof_to_test" or (args.algorithm == "GA" and args.test and not args.train):
                                if attr != "play_against_yourself" or args.test:
                                    print(f"{attr.replace('_', ' ').capitalize()}: {value}")


def initialize_env(args):
    """Initialize the environment based on the chosen mode."""
    atari_game_module = importlib.import_module(f"pettingzoo.atari.{args.atari_game}")
    if args.env_mode == "AEC":
        env = atari_game_module.env(render_mode="human" if args.render else None, obs_type="grayscale_image")
        env = frame_skip_v0(env, 4)
        env = resize_v1(env, 84, 84)
        env = frame_stack_v1(env, 4)
        env = agent_indicator_v0(env)
    elif args.env_mode == "parallel":
        env = atari_game_module.parallel_env(render_mode="human" if args.render else None)
    else:
        raise ValueError("Invalid environment mode. Choose either 'AEC' or 'parallel'.")
    env.reset(seed=1938214)
    return env

def main():
    
    args = parse_arguments()

    env = initialize_env(args)

    hof = []


    print("\nHyperparameters and Parameters:")
    args = Args(args)
    args.print_attributes(args)
    print("\n")


    if args.train:

        print("Starting Training...")

        agent = env.agents[0]
        input_channels = env.observation_space(agent).shape[-1]
        num_actions = env.action_space(agent).n

        # Train agent using the selected algorithm
        if args.algorithm == "GA":
            hof = genetic_algorithm_train(env, agent, args)
        elif args.algorithm == "ES":
            agent = evolution_strategy_train(env, agent, args)
        else:
            print("Unknown algorithm. Exiting.")
            return

        print("Training completed.")
    env.close()


    if args.test:

        if args.algorithm == "ES":
           
            if args.ES_model_to_test is not None and not os.path.exists(args.ES_model_to_test):
                print(f"Error: Model file {args.ES_model_to_test} not found.")
                return
                

            print(f"Loading model from {args.ES_model_to_test} for testing...")
            agent = torch.load(args.ES_model_to_test)
            total_rewards = 0
            test_episodes = 10
            for episode in range(10):

                if args.play_against_yourself:
                    reward, _, timesteps = play_game(env=env, player1=agent.model,
                                                player2=agent.model, 
                                                args=args, eval=True)
                else:
                    reward, _, timesteps = play_game(env=env, player1=agent.model,
                                                player2=RandomPolicy(env.action_space(env.agents[0]).n), 
                                                args=args, eval=True)

            total_rewards += reward

            avg_reward = total_rewards / test_episodes
            print(f"\n Average Reward of Best Model over {test_episodes} Episodes: {avg_reward}")




        else: # args.algorithm == "GA"
            
            # TODO: TO TEST
            
            hof = load_hof(args.model_to_test)
            env = initialize_env(args)  

            best_model = Agent(args.input_channels, n_actions=env.action_space(env.agents[0]).n, precision=args.precision)
            best_model.set_weights(hof[-1])

            total_rewards = 0
            test_episodes = 10
            for episode in range(10):
                reward, _, timesteps = play_game(env=env, player1=best_model.model,
                                            player2=RandomPolicy(env.action_space(env.agents[0]).n), 
                                            args=args, eval=True)
            total_rewards += reward

            avg_reward = total_rewards / test_episodes
            print(f"\n Average Reward of Best Model over {test_episodes} Episodes: {avg_reward}")


if __name__ == "__main__":
    main()
