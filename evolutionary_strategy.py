import torch
import numpy as np
from deepqn import DeepQN
from genetic_algorithm import play_game, RandomPolicy, plot_rewards
from agent import Agent
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def get_numpy_dtype(precision):
    """Helper function to get numpy dtype from string."""
    if precision == "float16":
        return np.float16
    elif precision == "float32":
        return np.float32
    else:
        # If a different precision is needed, add it here.
        # Default to float32 if unsupported precision is given
        return np.float32


def create_es_model_dir(args):
    """Create a directory for saving models based on hyperparameters."""
    dir_name = f"ES_models/gens{args.generations}_pop{args.population}_hof{args.hof_size}_game{args.atari_game}_mut{args.mutation_power}_adaptive{args.adaptive}_lr{args.learning_rate}_tslimit{args.max_timesteps_per_episode}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def save_model(obj, file_path):
    """Save any object to a file."""
    torch.save(obj, file_path)

def evaluate_current_weights(agent, env, args):
    """Evaluate the current weights against a random policy."""
    total_reward = 0
    for _ in tqdm(range(10), desc="Evaluating weights", leave=False):  # Evaluate over multiple episodes
        reward, _, ts = play_game(env=env, player1=agent.model,
                                    player2=RandomPolicy(env.action_space(env.agents[0]).n),
                                    args=args, eval=True)
        total_reward += reward
    return total_reward / 10


def mutate_weights(env, base_weights, args):
    """Apply Gaussian noise to the weights."""
    np_dtype = get_numpy_dtype(args.precision)
    base_weights = base_weights.astype(np_dtype)
    agent = Agent(input_channels=args.input_channels, n_actions=env.action_space(env.agents[0]).n, precision=args.precision)
    agent.model.set_weights_ES(flat_weights=base_weights, args=args)
    mutated_agent = Agent(input_channels=args.input_channels, n_actions=env.action_space(env.agents[0]).n, precision=args.precision)
    mutated_agent.model.set_weights_ES(flat_weights=base_weights, args=args)
    perturbations = mutated_agent.mutate_ES(args)
    perturbations = perturbations.astype(np_dtype)
   
    _, mutated_agent_reward1, _ = play_game(env=env, player1=agent.model, player2=mutated_agent.model, args=args)
    mutated_agent_reward2, _, _ = play_game(env=env, player1=mutated_agent.model, player2=agent.model, args=args)
    
    total_reward = np.mean([mutated_agent_reward1, mutated_agent_reward2]).astype(np_dtype)

    return total_reward, perturbations


def compute_weight_update(noises, rewards, args):
    """Compute the weight update based on the rewards and noises."""
    np_dtype = get_numpy_dtype(args.precision)
    noises = np.array(noises, dtype=np_dtype)
    rewards = np.array(rewards, dtype=np_dtype)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) if np.std(rewards) > 0 else np_dtype(1.0)
    normalized_rewards = (rewards - mean_reward) / std_reward

    weights_update = (args.learning_rate / (len(noises) * args.mutation_power)) * np.dot(np.array(noises).T, normalized_rewards)

    weights_udpate = weights_update.astype(np_dtype)

    return weights_update

def evolution_strategy_train(env, agent, args):
    """Train the agent using Evolution Strategies with Elitism and Hall of Fame."""


    # Initialize directories and variables for saving
    model_dir = create_es_model_dir(args)
    agent_file = os.path.join(model_dir, "agent.pth")
    rewards_plot_file = os.path.join(model_dir, "rewards_plot.png")

    agent = Agent(input_channels=args.input_channels, n_actions=env.action_space(env.agents[0]).n, precision=args.precision)
    
    np_dtype = get_numpy_dtype(args.precision)
    
    base_weights = agent.model.get_perturbable_weights().astype(np_dtype)

    rewards_over_generations = []

    for gen in tqdm(range(args.generations), desc="Training Generations"):
        noises = []
        rewards = []

        # Step 1: Generate population by mutating weights
        for _ in tqdm(range(args.population), desc=f"Generation {gen} - Mutating", leave=False):
            total_reward, noise = mutate_weights(env, base_weights, args)
            noises.append(noise)
            rewards.append(total_reward)

      
        weight_update = compute_weight_update(noises, rewards, args)
        base_weights = base_weights + weight_update
        agent.model.set_weights_ES(flat_weights=base_weights, args=args)

        evaluation_reward = evaluate_current_weights(agent, env, args)
        rewards_over_generations.append(evaluation_reward)


        # Adaptive mutation power
        if args.adaptive:
            args.mutation_power = max(0.01, args.mutation_power * 0.95)  # Reduce over generations

        # Save agent and plot rewards at each generation
        save_model(agent, agent_file)
        
        average_window=10
        plot_rewards(rewards_over_generations, rewards_plot_file, window=average_window)



    return agent
