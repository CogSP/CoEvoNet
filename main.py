import argparse
import torch
from agent import Agent
from hyperparameters import Hyperparameters
from evolutionary_algorithms import genetic_algorithm_train, evolution_strategy_train
from multiagentenvironment import MultiAgentEnvironment # TODO: PETTINGZOO ENVIRONMENT TO TEST


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a DNN agent using GA or ES")
    parser.add_argument("--algorithm", type=str, choices=["GA", "ES"], default="GA", 
                        help="Choose the algorithm to use: GA (Genetic Algorithm) or ES (Evolution Strategies)")
    parser.add_argument("--generations", type=int, default=100, 
                        help="Number of generations for the evolutionary process")
    parser.add_argument("--population", type=int, default=200, 
                        help="Population size for the evolutionary algorithm")
    parser.add_argument("--noise_std", type=float, default=0.005, 
                        help="Noise standard deviation for weight mutation")
    parser.add_argument("--hof_size", type=int, default=10, 
                        help="Size of the Hall of Fame")
    args = parser.parse_args()

    # Initialize environment
    env = MultiAgentEnvironment()  #TODO: REPLACE WITH PETTINGZOO
    input_channels = env.observation_space.shape[0]  # E.g., 4 for a stack of 4 frames
    n_actions = env.action_space.n  # Number of possible actions

    # Initialize agent
    agent = Agent(input_channels, n_actions)

    # Initialize hyperparameters
    hyperparams = Hyperparameters(algo=args.algorithm)
    hyperparams.population_size = args.population
    hyperparams.noise_std = args.noise_std

    # Set max generations
    global MAX_GENERATIONS
    MAX_GENERATIONS = args.generations

    # Train agent using the selected algorithm
    if args.algorithm == "GA":
        print("Training using Genetic Algorithm (GA)...")
        genetic_algorithm_train(env, agent, hyperparams, MAX_GENERATIONS)
    elif args.algorithm == "ES":
        print("Training using Evolution Strategies (ES)...")
        evolution_strategy_train(env, agent, hyperparams, MAX_GENERATIONS)
    else:
        print("Unknown algorithm. Exiting.")
        return

    print("Training completed.")

if __name__ == "__main__":
    main()
