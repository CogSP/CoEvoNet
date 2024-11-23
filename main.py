import argparse
import importlib # to import dinamically the Atari Games
import torch
from agent import Agent
from hyperparameters import Hyperparameters
from evolutionary_algorithms import genetic_algorithm_train, evolution_strategy_train
from pettingzoo.atari import boxing_v2
from pettingzoo.atari import pong_v3


MAX_TIMESTEPS_PER_EPISODE = 500
TRAIN = True

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
    parser.add_argument("--atari_game", type=str, choices=["boxing_v2", "pong_v3"], default="pong_v3",
                        help="Choose the Atari games to test with")
    args = parser.parse_args()


    atari_game_module = importlib.import_module(f"pettingzoo.atari.{args.atari_game}")
    env = atari_game_module.env(render_mode="human")
    env.reset(seed=42)

    if TRAIN == True:

        agent = env.agents[0]
        print(f"agent: {agent}")
        input_channels = env.observation_space(agent).shape[-1]
        num_actions = env.action_space(agent).n
        
        # Initialize agent
        #agent = Agent(input_channels, num_actions)

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
