import torch
import numpy as np
from utils.game_logic_functions import play_game, create_agent, diversity_penalty
from utils.utils_policies import RandomPolicy
from utils.utils_pth_and_plots import plot_experiment_metrics, plot_weights_logging, save_model
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

def evaluate_current_weights(agent_0, agent_1, adversary, env, args):
    
    total_reward_agent_0 = 0
    total_reward_agent_1 = 0
    total_reward_adversary = 0

    for _ in tqdm(range(10), desc="Evaluating weights", leave=False): 

        rw_agent_0, rw_agent_1, rw_adversary = play_game(env=env, player1=agent_0.model,
                                        player2=agent_1.model, adversary=adversary.model,
                                        args=args, eval=True)

        total_reward_agent_0 += rw_agent_0
        total_reward_agent_1 += rw_agent_1
        total_reward_adversary += rw_adversary
            

        """
        if role == "agent_0":
            rw_agent_0, _, _ = play_game(env=env, player1=agent_0.model,
                                        player2=agent_1.model, adversary=adversary.model,
                                        args=args, eval=True)
            total_reward += rw_agent_0

        if role == "agent_1":
            _, rw_agent_1, _ = play_game(env=env, player1=agent_0.model,
                                        player2=agent_1.model, adversary=adversary.model,
                                        args=args, eval=True)
            total_reward += rw_agent_1

        if role == "adversary_0":
            _, _, rw_adversary = play_game(env=env, player1=agent_0.model,
                                        player2=agent_1.model, adversary=adversary.model,
                                        args=args, eval=True)
            total_reward += rw_adversary
        """

    return total_reward_agent_0 / 10, total_reward_agent_1 / 10, total_reward_adversary / 10



def mutate_weights(env, agent_0, agent_1, adversary, args, role, step, weights_logging_agent_0, weights_logging_agent_1, weights_logging_adversary):
    """Apply Gaussian noise to the weights."""

    if role == "agent_0":

        if args.debug:
            print("\n mutating agent_0:")
        
        np_dtype = get_numpy_dtype(args.precision)
        mutated_agent = agent_0.clone(env, args, role="agent_0")
        perturbations = mutated_agent.mutate_ES(args, role="agent_0", step=step, weights_logging_agent_0=weights_logging_agent_0, weights_logging_agent_1=weights_logging_agent_1, weights_logging_adversary=weights_logging_adversary)
        perturbations = perturbations.astype(np_dtype)
        mutated_weights = mutated_agent.model.get_weights_ES()

        mutated_agent_0_reward, _, _ = play_game(env=env, player1=mutated_agent.model, player2=agent_1.model, adversary=adversary.model, args=args)
        
        # mutated_agent_0_reward = mutated_agent_0_reward.astype(np_dtype)

        return mutated_agent_0_reward, perturbations, mutated_weights

    elif role == "agent_1":

        if args.debug:
            print("\n mutating agent_1:")
        
        np_dtype = get_numpy_dtype(args.precision)
        mutated_agent = agent_1.clone(env, args, role="agent_1")
        perturbations = mutated_agent.mutate_ES(args, role="agent_1", step=step, weights_logging_agent_0=weights_logging_agent_0, weights_logging_agent_1=weights_logging_agent_1, weights_logging_adversary=weights_logging_adversary)
        perturbations = perturbations.astype(np_dtype)
        mutated_weights = mutated_agent.model.get_weights_ES()

        _, mutated_agent_1_reward, _ = play_game(env=env, player1=agent_0.model, player2=mutated_agent.model, adversary=adversary.model, args=args)
        
        return mutated_agent_1_reward, perturbations, mutated_weights


    elif role == "adversary_0":

        if args.debug:
            print("\n mutating adversary:")
        

        np_dtype = get_numpy_dtype(args.precision)
        mutated_agent = adversary.clone(env, args, role="adversary_0")
        perturbations = mutated_agent.mutate_ES(args, role="adversary_0", step=step, weights_logging_agent_0=weights_logging_agent_0, weights_logging_agent_1=weights_logging_agent_1, weights_logging_adversary=weights_logging_adversary)
        perturbations = perturbations.astype(np_dtype)
        mutated_weights = mutated_agent.model.get_weights_ES()

        _, _, mutated_adversary_reward = play_game(env=env, player1=agent_0.model, player2=agent_1.model, adversary=mutated_agent.model, args=args)

        if args.debug:  
            print(f"\n mutated_adversary_reward = {mutated_adversary_reward}")

        return mutated_adversary_reward, perturbations, mutated_weights



def compute_weight_update(noises, rewards, args, role, individual_weights=None, population_weights=None):
    """Compute the weight update based on the rewards and noises."""
    np_dtype = get_numpy_dtype(args.precision)
    noises = np.array(noises, dtype=np_dtype)
    rewards = np.array(rewards, dtype=np_dtype)
        
    diversity = None
    if args.fitness_sharing:
        diversity = diversity_penalty(individual_weights=individual_weights, population_weights=population_weights, args=args)
        fitness = rewards / (1 + diversity)
    else:
        fitness = rewards

    #mean_fitness = np.mean(fitness)
    #std_fitness = np.std(fitness) if np.std(fitness) > 0 else np_dtype(1.0)
    #normalized_fitness = (fitness - mean_fitness) / std_fitness

    if role == "agent_0":
        mutation_power = args.mutation_power_agent_0
    elif role == "agent_1":
        mutation_power = args.mutation_power_agent_1
    elif role == "adversary_0":
        mutation_power = args.mutation_power_adversary

    weights_update = (args.learning_rate / (len(noises) * mutation_power)) * np.dot(np.array(noises).T, fitness)

    weights_update = weights_update.astype(np_dtype)

    return weights_update, diversity


def evolution_strategy_train(env, args, output_dir):
    """Train the agent using Evolution Strategies with Elitism and Hall of Fame."""

    agent_0_file = os.path.join(output_dir, "agent_0.pth")
    agent_1_file = os.path.join(output_dir, "agent_1.pth")
    adversary_file = os.path.join(output_dir, "adversary.pth")

    results_plot_file_agent_0 = os.path.join(output_dir, "results_plot_file_agent_0.png")
    results_plot_file_agent_1 = os.path.join(output_dir, "results_plot_file_agent_1.png")
    results_plot_file_adversary = os.path.join(output_dir, "results_plot_file_adversary.png")
    weights_results_plot_file = os.path.join(output_dir, "weights_results_plot_file.png")

    agent_0 = create_agent(env, args, role="agent_0")
    agent_1 = create_agent(env, args, role="agent_1")
    adversary = create_agent(env, args, role="adversary_0")

    total_params = sum(p.numel() for p in agent_0.model.parameters()) 
    print(f'\nNumber of parameters for agent_0 network: {total_params}')
    total_params = sum(p.numel() for p in agent_1.model.parameters()) 
    print(f'\nNumber of parameters for agent_1 network: {total_params}')
    total_params = sum(p.numel() for p in adversary.model.parameters()) 
    print(f'\nNumber of parameters for adversary network: {total_params}')

    np_dtype = get_numpy_dtype(args.precision)
    base_weights_agent_0 = agent_0.model.get_perturbable_weights().astype(np_dtype)
    base_weights_agent_1 = agent_1.model.get_perturbable_weights().astype(np_dtype)
    base_weights_adversary = adversary.model.get_perturbable_weights().astype(np_dtype)
    
    rewards_over_generations_agent_0 = []
    rewards_over_generations_agent_1 = []
    rewards_over_generations_adversary = []
    weights_logging_agent_0 = []
    weights_logging_agent_1 = []
    weights_logging_adversary = []


    if args.fitness_sharing:
        diversity_over_generations_agent_0 = []
        fitness_over_generations_agent_0 = []
        diversity_over_generations_agent_1 = []
        fitness_over_generations_agent_1 = []
        diversity_over_generations_adversary = []
        fitness_over_generations_adversary = []
    else:
        diversity_over_generations_agent_0 = None
        fitness_over_generations_agent_0 = None
        diversity_over_generations_agent_1 = None
        fitness_over_generations_agent_1 = None
        diversity_over_generations_adversary = None
        fitness_over_generations_adversary = None


    if args.adaptive:
        mutation_power_history_agent_0 = [args.mutation_power_agent_0]
        mutation_power_history_agent_1 = [args.mutation_power_agent_1]
        mutation_power_history_adversary = [args.mutation_power_adversary]
    else:
        mutation_power_history_agent_0 = None
        mutation_power_history_agent_1 = None
        mutation_power_history_adversary = None


    if args.early_stopping:
        best_reward_agent_0 = -float("inf")
        no_improvement_count_agent_0 = 0
        best_reward_agent_1 = -float("inf")
        no_improvement_count_agent_1 = 0
        best_reward_adversary = -float("inf")
        no_improvement_count_adversary = 0
    

    for gen in tqdm(range(args.generations), desc="Training Generations"):
        
        noises_agent_0 = []
        rewards_agent_0 = []
        population_weights_agent_0 = []

        noises_agent_1 = []
        rewards_agent_1 = []
        population_weights_agent_1 = []

        noises_adversary = []
        rewards_adversary = []
        population_weights_adversary = []

        for _ in tqdm(range(args.population), desc=f"Generation {gen} - Mutating", leave=False):

            total_reward_agent_0, noise_agent_0, mutated_weights_agent_0 = mutate_weights(env, agent_0, agent_1, adversary, args, "agent_0", gen, weights_logging_agent_0, weights_logging_agent_1, weights_logging_adversary)
            noises_agent_0.append(noise_agent_0)
            rewards_agent_0.append(total_reward_agent_0)
            population_weights_agent_0.append(mutated_weights_agent_0)

            total_reward_agent_1, noise_agent_1, mutated_weights_agent_1 = mutate_weights(env, agent_0, agent_1, adversary, args, "agent_1", gen, weights_logging_agent_0, weights_logging_agent_1, weights_logging_adversary)
            noises_agent_1.append(noise_agent_1)
            rewards_agent_1.append(total_reward_agent_1)
            population_weights_agent_1.append(mutated_weights_agent_1)

            total_reward_adversary, noise_adversary, mutated_weights_adversary = mutate_weights(env, agent_0, agent_1, adversary, args, "adversary_0", gen, weights_logging_agent_0, weights_logging_agent_1, weights_logging_adversary)
            noises_adversary.append(noise_adversary)
            rewards_adversary.append(total_reward_adversary)
            population_weights_adversary.append(mutated_weights_adversary)

      
        # compute the update for all of the three agents
        weight_update_agent_0, diversity_agent_0 = compute_weight_update(noises_agent_0, rewards_agent_0, args, role="agent_0", individual_weights=base_weights_agent_0, population_weights=population_weights_agent_0)
        weight_update_agent_1, diversity_agent_1 = compute_weight_update(noises_agent_1, rewards_agent_1, args, role="agent_1", individual_weights=base_weights_agent_1, population_weights=population_weights_agent_1)
        weight_update_adversary, diversity_adversary = compute_weight_update(noises_adversary, rewards_adversary, args, role="adversary_0", individual_weights=base_weights_adversary, population_weights=population_weights_adversary)
        
        base_weights_agent_0 = base_weights_agent_0 + weight_update_agent_0
        base_weights_agent_1 = base_weights_agent_1 + weight_update_agent_1
        base_weights_adversary = base_weights_adversary + weight_update_adversary
        
        agent_0.model.set_weights_ES(flat_weights=base_weights_agent_0, args=args)
        agent_1.model.set_weights_ES(flat_weights=base_weights_agent_1, args=args)
        adversary.model.set_weights_ES(flat_weights=base_weights_adversary, args=args)

        """
        evaluation_reward_agent_0 = evaluate_current_weights(agent_0, agent_1, adversary, env, args, role="agent_0")
        evaluation_reward_agent_1 = evaluate_current_weights(agent_0, agent_1, adversary, env, args, role="agent_1")
        evaluation_reward_adversary = evaluate_current_weights(agent_0, agent_1, adversary, env, args, role="adversary_0")
        """
        evaluation_reward_agent_0, evaluation_reward_agent_1, evaluation_reward_adversary = evaluate_current_weights(agent_0, agent_1, adversary, env, args) 

        rewards_over_generations_agent_0.append(evaluation_reward_agent_0)
        rewards_over_generations_agent_1.append(evaluation_reward_agent_1)
        rewards_over_generations_adversary.append(evaluation_reward_adversary)

        if args.fitness_sharing:
            diversity_over_generations_agent_0.append(diversity_agent_0)
            evaluation_fitness_agent_0 = evaluation_reward_agent_0 / (1 + diversity_agent_0)
            fitness_over_generations_agent_0.append(evaluation_fitness_agent_0)

            diversity_over_generations_agent_1.append(diversity_agent_1)
            evaluation_fitness_agent_1 = evaluation_reward_agent_1 / (1 + diversity_agent_1)
            fitness_over_generations_agent_1.append(evaluation_fitness_agent_1)

            diversity_over_generations_adversary.append(diversity_adversary)
            evaluation_fitness_adversary = evaluation_reward_adversary / (1 + diversity_adversary)
            fitness_over_generations_adversary.append(evaluation_fitness_adversary)

        # Dynamic Mutation Power via Reward Feedback
        if args.adaptive:

            if gen > 10 and np.mean(rewards_over_generations_agent_0[-10:]) < np.mean(rewards_over_generations_agent_0[-20:-10]):
                args.mutation_power_agent_0 = min(args.mutation_power_agent_1 * 1.2, args.max_mutation_power)
            else:
                args.mutation_power_agent_0 = max(args.mutation_power_agent_0 * 0.95, args.min_mutation_power)

            mutation_power_history_agent_0.append(args.mutation_power_agent_0)


            if gen > 10 and np.mean(rewards_over_generations_agent_1[-10:]) < np.mean(rewards_over_generations_agent_1[-20:-10]):
                args.mutation_power_agent_1 = min(args.mutation_power_agent_1 * 1.2, args.max_mutation_power)
            else:
                args.mutation_power_agent_1 = max(args.mutation_power_agent_1 * 0.95, args.min_mutation_power)

            mutation_power_history_agent_1.append(args.mutation_power_agent_1)



            if gen > 10 and np.mean(rewards_over_generations_adversary[-10:]) < np.mean(rewards_over_generations_adversary[-20:-10]):
                args.mutation_power_adversary = min(args.mutation_power_adversary * 1.2, args.max_mutation_power)
            else:
                args.mutation_power_adversary = max(args.mutation_power_adversary * 0.95, args.min_mutation_power)

            mutation_power_history_adversary.append(args.mutation_power_adversary)


        
        # Check for improvement for early stopping
        if args.early_stopping:
            # If the improvement is larger than min_delta, reset no_improvement_count
            if evaluation_reward_agent_0 > best_reward_agent_0 + args.min_delta:
                best_reward_agent_0 = evaluation_reward_agent_0
                no_improvement_count_agent_0 = 0
            else:
                no_improvement_count_agent_0 += 1
        
            if evaluation_reward_agent_1 > best_reward_agent_1 + args.min_delta:
                best_reward_agent_1 = evaluation_reward_agent_1
                no_improvement_count_agent_1 = 0
            else:
                no_improvement_count_agent_1 += 1
        

            if evaluation_reward_adversary > best_reward_adversary + args.min_delta:
                best_reward_adversary = evaluation_reward_adversary
                no_improvement_count_adversary = 0
            else:
                no_improvement_count_adversary += 1
        
            # If no improvement for 'patience' generations, stop training
            if no_improvement_count_agent_0 >= args.patience:
                print(f"Early stopping triggered at generation {gen} for agent_0. Best reward: {best_reward_agent_0}")
                break
        
            # If no improvement for 'patience' generations, stop training
            if no_improvement_count_agent_1 >= args.patience:
                print(f"Early stopping triggered at generation {gen} for agent_1. Best reward: {best_reward_agent_1}")
                break
            
            # If no improvement for 'patience' generations, stop training
            if no_improvement_count_adversary >= args.patience:
                print(f"Early stopping triggered at generation {gen} for adversary. Best reward: {best_reward_adversary}")
                break

        # Save agent and plot rewards at each generation
        if args.save:
            save_model(agent_0, agent_0_file)
            save_model(agent_1, agent_1_file)
            save_model(adversary, adversary_file)
        

        plot_experiment_metrics(rewards=rewards_over_generations_agent_0, 
                                mutation_power_history=mutation_power_history_agent_0, 
                                fitness=fitness_over_generations_agent_0, 
                                diversity=diversity_over_generations_agent_0,
                                file_path=results_plot_file_agent_0,
                                args=args
                                )


        plot_experiment_metrics(rewards=rewards_over_generations_agent_1, 
                                mutation_power_history=mutation_power_history_agent_1, 
                                fitness=fitness_over_generations_agent_1, 
                                diversity=diversity_over_generations_agent_1,
                                file_path=results_plot_file_agent_1,
                                args=args
                                )


        plot_experiment_metrics(rewards=rewards_over_generations_adversary, 
                                mutation_power_history=mutation_power_history_adversary, 
                                fitness=fitness_over_generations_adversary, 
                                diversity=diversity_over_generations_adversary,
                                file_path=results_plot_file_adversary,
                                args=args
                                )


        plot_weights_logging(weights_results_plot_file, weights_logging_agent_0, weights_logging_agent_1, weights_logging_adversary)


    return agent_0, agent_1, adversary
