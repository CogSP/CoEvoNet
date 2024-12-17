import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from utils.game_logic_functions import create_agent


def load_agent_for_testing(args, env=None):

    if args.algorithm == "GA":

        if args.GA_hof_to_test is None:
            print(f"Error: HoF file not specified. Please specify the HoF to test")
            return
            
        if not os.path.exists(args.GA_hof_to_test):
            print(f"Error: Model file {args.GA_hof_to_test} not found.")
            return

        print(f"Loading HoF from {args.GA_hof_to_test} for testing...")
        hof = torch.load(args.GA_hof_to_test)
        best_model_weights = hof[-1].get_weights()
        agent = create_agent(env, args)
        agent.set_weights(best_model_weights)

        return agent

    elif args.algorithm == "ES":

        if args.ES_model_to_test_agent_0 is None:
            raise ValueError(f"Error: Model file for agent_0 not specified. Please specify the agent to test")
            return
        
        if args.ES_model_to_test_agent_1 is None:
            raise ValueError(f"Error: Model file for agent_1 not specified. Please specify the agent to test")
            return
        
        if args.ES_model_to_test_adversary_0 is None:
            raise ValueError(f"Error: Model file for adversary_0 not specified. Please specify the agent to test")
            return

        if not os.path.exists(args.ES_model_to_test_agent_0):
            raise ValueError(f"Error: Model file {args.ES_model_to_test_agent_0} not found.")
            return

        if not os.path.exists(args.ES_model_to_test_agent_1):
            raise ValueError(f"Error: Model file {args.ES_model_to_test_agent_1} not found.")
            return

        if not os.path.exists(args.ES_model_to_test_adversary_0):
            raise ValueError(f"Error: Model file {args.ES_model_to_test_adversary_0} not found.")
            return

        print(f"Loading Agents for testing...")
        agent_0 = torch.load(args.ES_model_to_test_agent_0)
        agent_1 = torch.load(args.ES_model_to_test_agent_1)
        adversary = torch.load(args.ES_model_to_test_adversary_0)

        return agent_0, agent_1, adversary



def save_model(obj, file_path):
    """Save any object to a file."""
    torch.save(obj, file_path)


def create_output_dir(args):
    """Create a directory for saving models and plots."""

    # TODO: add mutation power

    dir_name = f"{args.algorithm}_models/gens{args.generations}_pop{args.population}_hof{args.hof_size}_game{args.game}_tslimit{args.max_timesteps_per_episode}_fitness-sharing{args.fitness_sharing}_adaptive{args.adaptive}"
    if args.adaptive:
        dir_name += f"max_mutation{args.max_mutation_power}_min_mutation{args.min_mutation_power}"
    if args.algorithm == "ES":
        dir_name += f"_lr{args.learning_rate}"
    #if args.game == "simple_adversary_v3":
    #    dir_name += f"adversary{args.adversary}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


import matplotlib.pyplot as plt

def plot_weights_logging(file_path, weights_logging_agent_0, weights_logging_agent_1, weights_logging_adversary):
    """
    Plots the mean, min, max, and std deviation values of weights over steps for three agents/adversaries.

    Args:
        file_path (str): The base file path for saving plots.
        weights_logging_agent_0 (list of dict): Data for agent 0.
        weights_logging_agent_1 (list of dict): Data for agent 1.
        weights_logging_adversary (list of dict): Data for adversary.
    """

    def plot_data(data, title, save_path):
        """
        Helper function to plot data for a single agent/adversary.
        """
        steps = [entry["step"] for entry in data]
        means = [entry["mean"] for entry in data]
        mins = [entry["min"] for entry in data]
        maxs = [entry["max"] for entry in data]
        stds = [entry["std"] for entry in data]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, means, label="Mean", marker='o', linestyle='-')
        plt.plot(steps, mins, label="Min", marker='x', linestyle='--')
        plt.plot(steps, maxs, label="Max", marker='s', linestyle=':')
        plt.fill_between(steps, 
                         [mean - std for mean, std in zip(means, stds)],
                         [mean + std for mean, std in zip(means, stds)],
                         color='gray', alpha=0.2, label="Mean ± Std Dev")

        # Add labels, title, and legend
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel("Weight Value")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # Plot for each agent/adversary
    plot_data(weights_logging_agent_0, 
              "Weight Statistics Over Steps (Agent 0)", 
              f"{file_path}_agent_0.png")
    plot_data(weights_logging_agent_1, 
              "Weight Statistics Over Steps (Agent 1)", 
              f"{file_path}_agent_1.png")
    plot_data(weights_logging_adversary, 
              "Weight Statistics Over Steps (Adversary)", 
              f"{file_path}_adversary.png")




def plot_experiment_metrics(rewards=None, mutation_power_history=None, fitness=None, diversity=None, file_path="experiment_metrics.png", args=None):
    """
    Plot metrics such as rewards, mutation power, fitness, and diversity in a single function.
    
    Args:
        rewards (list): List of total rewards per generation.
        mutation_power_history (list): List of mutation power values over generations.
        fitness (list): List of fitness values across generations.
        diversity (list): List of diversity metrics over generations.
        file_path (str): Path to save the plot.
        args: Hyperparameters and config used in the experiment.
    """

    plt.figure(figsize=(24, 12))
    
    # Rewards Plot
    if rewards is not None:
        plt.subplot(2, 2, 1)
        plt.plot(rewards, marker='o', label="Total Reward")
        if len(rewards) >= args.average_window:
            moving_avg = [np.mean(rewards[max(0, i - args.average_window + 1):i + 1]) for i in range(len(rewards))]
            plt.plot(moving_avg, linestyle='-', label=f"Avg Reward (Last {args.average_window})")
        plt.title("Reward Progression Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)

    # Mutation Power Plot
    if mutation_power_history is not None:
        plt.subplot(2, 2, 2)
        generations = range(len(mutation_power_history))
        plt.plot(generations, mutation_power_history, label="Mutation Power", alpha=0.7)
        if len(mutation_power_history) >= args.average_window:
            moving_avg = [np.mean(mutation_power_history[max(0, i - args.average_window + 1):i + 1]) for i in range(len(mutation_power_history))]
            plt.plot(generations, moving_avg, label=f"Moving Avg (Last {args.average_window})", color="orange", linestyle='-')
        plt.axhline(args.max_mutation_power, color='r', linestyle='--', label="Max Mutation Power")
        plt.axhline(args.min_mutation_power, color='b', linestyle='--', label="Min Mutation Power")
        plt.title("Mutation Power Progression Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Mutation Power")
        plt.legend()
        plt.grid(True)

    # Fitness Plot
    if fitness is not None:
        plt.subplot(2, 2, 3)
        plt.plot(fitness, marker='x', label="Fitness")
        if len(fitness) >= args.average_window:
            moving_avg = [np.mean(fitness[max(0, i - args.average_window + 1):i + 1]) for i in range(len(fitness))]
            plt.plot(moving_avg, linestyle='-', label=f"Avg Fitness (Last {args.average_window})")
        plt.title("Fitness Progression Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)

    # Diversity Plot
    if diversity is not None:
        plt.subplot(2, 2, 4)
        plt.plot(diversity, marker='s', label="Diversity")
        if len(diversity) >= args.average_window:
            moving_avg = [np.mean(diversity[max(0, i - args.average_window + 1):i + 1]) for i in range(len(diversity))]
            plt.plot(moving_avg, linestyle='-', label=f"Avg Diversity (Last {args.average_window})")
        plt.title("Diversity Progression Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.legend()
        plt.grid(True)

    # Add hyperparameters to the figure
    if args is not None:
        hyperparams = (
            f"Algorithm: {args.algorithm}\n"
            f"Generations: {args.generations}\n"
            f"Population: {args.population}\n"
            f"HoF Size: {args.hof_size if hasattr(args, 'hof_size') else 'N/A'}\n"
            f"Game: {args.game}\n"
        )
        if args.adaptive:
            hyperparams += "Mutation Power: adaptive\n"
        else:
            hyperparams += f"Mutation Power: {args.mutation_power}\n"

        if args.algorithm == 'ES':
            hyperparams += f"Learning Rate: {args.learning_rate}\n"

        if args.fitness_sharing:
            hyperparams += "Fitness Sharing Enabled\n"

        if args.early_stopping:
            hyperparams += "Early Stopping Enabled\n"
            hyperparams += f"Patience: {args.patience}\n"
            hyperparams += f"Min Delta: {args.min_delta}\n"

        #if args.game == "simple_adversary_v3":
        #    hyperparams += f"Adversary: {args.adversary}"

        if args.game != 'simple_adversary_v3':
            hyperparams += f"Max Timesteps: {args.max_timesteps_per_episode}\n"
            hyperparams += f"Max Evaluation Steps: {args.max_evaluation_steps}\n"

        if args.algorithm == 'GA':
            hyperparams += f"Elites: {args.elites_number}\n"

        plt.gcf().text(0.02, 0.5, hyperparams, fontsize=18, ha='left', va='center', bbox=dict(boxstyle="round,pad=1", alpha=0.5))

    plt.tight_layout(rect=[0.2, 0, 1, 1])  # Adjust layout to leave more space for the left-side text box
    plt.savefig(file_path)
    plt.close()