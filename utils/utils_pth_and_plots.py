import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def load_agent_for_testing(args):

    if args.algorithm == "GA":

        if args.GA_hof_to_test is None:
            print(f"Error: HoF file not specified. Please specify the HoF to test")
            return
            
        if not os.path.exists(args.GA_hof_to_test):
            print(f"Error: Model file {args.GA_hof_to_test} not found.")
            return

        print(f"Loading HoF from {args.GA_hof_to_test} for testing...")
        hof = torch.load(file_path)
        agent = create_agent(env, args)
        agent.set_weights(hof[-1])

        return agent

    elif args.algorithm == "ES":

        if args.ES_model_to_test is None:
            print(f"Error: Model file not specified. Please specify the agent to test")
            return
            
        if not os.path.exists(args.ES_model_to_test):
            print(f"Error: Model file {args.ES_model_to_test} not found.")
            return

        print(f"Loading Agent from {args.ES_model_to_test} for testing...")
        agent = torch.load(args.ES_model_to_test)

        return agent



def save_model(obj, file_path):
    """Save any object to a file."""
    torch.save(obj, file_path)


def create_output_dir(args):
    """Create a directory for saving models and plots."""
    dir_name = f"{args.algorithm}_models/gens{args.generations}_pop{args.population}_hof{args.hof_size}_game{args.game}_mut{args.mutation_power}_adaptive{args.adaptive}_tslimit{args.max_timesteps_per_episode}"
    if args.algorithm == "ES":
        dir_name += f"_lr{args.learning_rate}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def plot_rewards(rewards, file_path, window=10):
    """
    Plot rewards across generations and their moving average over a window.
    
    Args:
        rewards (list): List of total rewards per generation.
        file_path (str): Path to save the plot.
        window (int): Window size for calculating the moving average.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards
    plt.plot(rewards, marker='o', label="Total Reward")
    
    # Calculate moving average
    if len(rewards) >= window:
        moving_avg = [np.mean(rewards[max(0, i - window + 1):i + 1]) for i in range(len(rewards))]
        plt.plot(moving_avg, linestyle='-', label=f"Avg Reward (Last {window})")
    
    plt.title("Reward Progression Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.savefig(file_path)
    plt.close()
