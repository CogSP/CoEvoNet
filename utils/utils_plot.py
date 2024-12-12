import matplotlib.pyplot as plt
import numpy as np

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
