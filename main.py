import argparse
import importlib # to import dinamically the Atari Games
import torch
from agent import Agent
from genetic_algorithm import genetic_algorithm_train
from evolutionary_strategy import evolution_strategy_train


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
    parser.add_argument("--hof_size", type=int, default=5, 
                        help="Size of the Hall of Fame")
    parser.add_argument("--atari_game", type=str, choices=["boxing_v2", "pong_v3"], default="pong_v3",
                        help="Choose the Atari games to test with")
    parser.add_argument("--initial_mutation_power", type=float, default=0.05, 
                        help="Initial value for the mutation power, i.e. noise standard deviation for weight mutation")
    parser.add_argument("--learning_rate", type=float, default=0.1, 
                        help="Learning rate")
    parser.add_argument("--max_timesteps_per_episode", type=int, default=3, 
                        help="Total timesteps per episode")
    parser.add_argument("--max_evaluation_steps", type=int, default=3, 
                        help="Maximum steps for evaluation")
    parser.add_argument("--input_channels", type=int, default=3, 
                        help="Number of input channels for the observation")
    parser.add_argument("--elites_number", type=int, default=5, 
                        help="Number of elites for each generation")          
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode for detailed logging")
    parser.add_argument("--train", action="store_true",
                        help="Train the models or just test")

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

        # clearly, these are not hyperparam, but it's easy to have everything inside an object
        self.debug = args.debug
        self.train = args.train
        self.input_channels = args.input_channels


    def print_attributes(self): 
        # Print all attributes except `input_channel`
        attributes = vars(self)  # Get all attributes as a dictionary
        for attr, value in attributes.items():
            if attr != "input_channels":  # Skip `input_channel`
                print(f"{attr.replace('_', ' ').capitalize()}: {value}")


def main():
    
    args = parse_arguments()

    atari_game_module = importlib.import_module(f"pettingzoo.atari.{args.atari_game}")
    env = atari_game_module.env()
    env.reset(seed=42)

    if args.train:

        print("Starting Training...")

        agent = env.agents[0]
        input_channels = env.observation_space(agent).shape[-1]
        num_actions = env.action_space(agent).n

        print("\nHyperparameters and Parameters:")
        args = Args(args)
        args.print_attributes()
        print("\n")


        # Train agent using the selected algorithm
        if args.algorithm == "GA":
            genetic_algorithm_train(env, agent, args)
        elif args.algorithm == "ES":
            evolution_strategy_train(env, agent, args)
        else:
            print("Unknown algorithm. Exiting.")
            return

        print("Training completed.")


    # now we can test
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()

        env.step(action)
    env.close()


if __name__ == "__main__":
    main()
