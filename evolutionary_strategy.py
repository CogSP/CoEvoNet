import torch
import numpy as np
from utils.game_logic_functions import play_game, create_agent, diversity_penalty
from utils.utils_policies import RandomPolicy
from utils.utils_pth_and_plots import plot_experiment_metrics, save_model
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

    # if MPE, the RP is the adversarial against two instances of the genetic model
    # and will be instantiated inside play_game
    player2 = RandomPolicy(env.action_space(env.agents[0]).n) if args.game != "simple_adversary_v3" else agent.model

    for _ in tqdm(range(10), desc="Evaluating weights", leave=False): 
        reward, _ = play_game(env=env, player1=agent.model,
                                    player2=player2,
                                    args=args, eval=True)
        total_reward += reward
    return total_reward / 10


def mutate_weights(env, agent, args):
    """Apply Gaussian noise to the weights."""
    np_dtype = get_numpy_dtype(args.precision)
    mutated_agent = agent.clone(env, args)
    perturbations = mutated_agent.mutate_ES(args)
    perturbations = perturbations.astype(np_dtype)
    mutated_weights = mutated_agent.model.get_weights_ES()

    _, mutated_agent_reward1 = play_game(env=env, player1=agent.model, player2=mutated_agent.model, args=args)
    mutated_agent_reward2, _ = play_game(env=env, player1=mutated_agent.model, player2=agent.model, args=args)
    

    total_reward = np.mean([mutated_agent_reward1, mutated_agent_reward2]).astype(np_dtype)

    return total_reward, perturbations, mutated_weights


def compute_weight_update(noises, rewards, args, individual_weights=None, population_weights=None):
    """Compute the weight update based on the rewards and noises."""
    np_dtype = get_numpy_dtype(args.precision)
    noises = np.array(noises, dtype=np_dtype)
    rewards = np.array(rewards, dtype=np_dtype)
        
    diversity = None
    if args.fitness_sharing:
        diversity = diversity_penalty(individual_weights=individual_weights, population_weights=population_weights, args=args)
        # note that diversity is 1 when the sharing function is 0, so I removed the 1 + diversity for just a diversity denominator
        fitness = rewards / diversity
    else:
        fitness = rewards

    mean_fitness = np.mean(fitness)
    std_fitness = np.std(fitness) if np.std(fitness) > 0 else np_dtype(1.0)
    normalized_fitness = (fitness - mean_fitness) / std_fitness

    weights_update = (args.learning_rate / (len(noises) * args.mutation_power)) * np.dot(np.array(noises).T, normalized_fitness)

    weights_udpate = weights_update.astype(np_dtype)

    return weights_update, diversity


def evolution_strategy_train(env, agent, args, output_dir):
    """Train the agent using Evolution Strategies with Elitism and Hall of Fame."""

    agent_file = os.path.join(output_dir, "agent.pth")
    
    results_plot_file = os.path.join(output_dir, "results_plot_file.png")

    agent = create_agent(env, args)

    total_params = sum(p.numel() for p in agent.model.parameters()) 
    print(f'\nNumber of parameters of each network: {total_params}')
    
    np_dtype = get_numpy_dtype(args.precision)
    base_weights = agent.model.get_perturbable_weights().astype(np_dtype)
    
    rewards_over_generations = []

    if args.fitness_sharing:
        diversity_over_generations = []
        fitness_over_generations = []
    else:
        diversity_over_generations = None
        fitness_over_generations = None


    if args.adaptive:
        mutation_power_history = [args.mutation_power]
    else:
        mutation_power_history = None


    for gen in tqdm(range(args.generations), desc="Training Generations"):
        
        noises = []
        rewards = []
        population_weights = [base_weights]

        for _ in tqdm(range(args.population), desc=f"Generation {gen} - Mutating", leave=False):
            total_reward, noise, mutated_weights = mutate_weights(env, agent, args)
            noises.append(noise)
            rewards.append(total_reward)
            population_weights.append(mutated_weights)

      
        weight_update, diversity = compute_weight_update(noises, rewards, args, base_weights, population_weights)
        base_weights = base_weights + weight_update
        agent.model.set_weights_ES(flat_weights=base_weights, args=args)

        evaluation_reward = evaluate_current_weights(agent, env, args)
        rewards_over_generations.append(evaluation_reward)

        if args.fitness_sharing:
            diversity_over_generations.append(diversity)
            evaluation_fitness = evaluation_reward / diversity 
            fitness_over_generations.append(evaluation_fitness)

        # Dynamic Mutation Power via Reward Feedback
        if args.adaptive:

            if gen > 10 and np.mean(rewards_over_generations[-10:]) < np.mean(rewards_over_generations[-20:-10]):
                args.mutation_power = min(args.mutation_power * 1.2, args.max_mutation_power)
            else:
                args.mutation_power = max(args.mutation_power * 0.95, args.min_mutation_power)

            mutation_power_history.append(args.mutation_power)

        # Save agent and plot rewards at each generation
        if args.save:
            save_model(agent, agent_file)
        

        plot_experiment_metrics(rewards=rewards_over_generations, 
                                mutation_power_history=mutation_power_history, 
                                fitness=fitness_over_generations, 
                                diversity=diversity_over_generations,
                                file_path=results_plot_file,
                                args=args
                                )

    return agent
