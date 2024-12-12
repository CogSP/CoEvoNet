import torch
import numpy as np
from utils.game_logic_functions import play_game, create_agent
from utils.utils_policies import RandomPolicy
from utils.utils_pth_and_plots import plot_rewards, save_model
from agent import Agent
from tqdm import tqdm
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

def evaluate_current_weights(agent, env, args):
    """Evaluate the current weights ten times against a random policy."""
    total_reward = 0
    for _ in tqdm(range(10), desc="Evaluating weights", leave=False): 
        reward, _ = play_game(env=env, player1=agent.model,
                                    player2=RandomPolicy(env.action_space(env.agents[0]).n),
                                    args=args, eval=True)
        total_reward += reward
    return total_reward / 10


def mutate_weights(env, agent, args):
    """Apply Gaussian noise to the weights."""
    np_dtype = get_numpy_dtype(args.precision)
    mutated_agent = agent.clone(env, args)
    perturbations = mutated_agent.mutate_ES(args)
    perturbations = perturbations.astype(np_dtype)
   
    _, mutated_agent_reward1 = play_game(env=env, player1=agent.model, player2=mutated_agent.model, args=args)
    mutated_agent_reward2, _ = play_game(env=env, player1=mutated_agent.model, player2=agent.model, args=args)
    

    # TODO: why are we using the mean here and not the sum as in GA?
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

def evolution_strategy_train(env, agent, args, output_dir):
    """Train the agent using Evolution Strategies with Elitism and Hall of Fame."""


    # Initialize directories and variables for saving
    agent_file = os.path.join(output_dir, "agent.pth")
    rewards_plot_file = os.path.join(output_dir, "rewards_plot.png")

    agent = create_agent(env, args)

    total_params = sum(p.numel() for p in agent.model.parameters()) 
    print(f'\n Number of parameters of each network: {total_params}')
    
    np_dtype = get_numpy_dtype(args.precision)
    base_weights = agent.model.get_perturbable_weights().astype(np_dtype)
    
    rewards_over_generations = []

    for gen in tqdm(range(args.generations), desc="Training Generations"):
        noises = []
        rewards = []

        # Step 1: Generate population by mutating weights
        for _ in tqdm(range(args.population), desc=f"Generation {gen} - Mutating", leave=False):
            total_reward, noise = mutate_weights(env, agent, args)
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
        if args.save:
            save_model(agent, agent_file)
        
        average_window=50
        plot_rewards(rewards_over_generations, rewards_plot_file, window=average_window)

    return agent
