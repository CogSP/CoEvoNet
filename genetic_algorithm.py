import torch
import numpy as np
from utils.game_logic_functions import create_agent, play_game
from utils.utils_plot import plot_rewards
from utils.game_logic_functions import initialize_env
from utils.utils_policies import RandomPolicy
from agent import Agent
from tqdm import tqdm
import os


def create_ga_model_dir(args):
    """Create a directory for saving models based on hyperparameters."""
    dir_name = f"GA_models/gens{args.generations}_pop{args.population}_hof{args.hof_size}_game{args.game}_mut{args.mutation_power}_adaptive{args.adaptive}_tslimit{args.max_timesteps_per_episode}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def save_hof(hof, file_path):
    """ Save the Hall of Fame (HoF) to disk. """
    torch.save(hof, file_path)
    #print(f"Hall of Fame saved to {file_path}")


def save_elites(elites, file_path):
    """ Save the elites to disk. """
    elite_weights = [elite.get_weights() for elite in elites]
    torch.save(elite_weights, file_path)
    #print(f"Elite weights saved to {file_path}")


def load_hof(file_path, env, args=None):
    """ Load the Hall of Fame (HoF) from disk. """
    if os.path.exists(file_path):
        hof = torch.load(file_path)
        print(f"Hall of Fame loaded from {file_path}")
    else:
        print(f"No HoF file found at {file_path}. Returning initial list of weights.")
        hof = [create_agent(env, args) for _ in range(args.hof_size)]
    return hof

def load_elites(file_path, env=None, args=None):
    """ Load the elites from disk. """
    elites_agents = []
    if os.path.exists(file_path):
        elites_agents = torch.load(file_path)
        print(f"Elites loaded from {file_path}")
    else:
        print(f"No elites found at {file_path}. Returning initial list of weights.")
        elites_agents = [create_agent(env, args) for _ in range(args.hof_size)]

        for agent in elites_agents:
            if args.precision == "float16":
                agent.model.half()
            else:
                agent.model.float()

    return elites_agents



def evaluate_current_weights(best_agent, env, args):
    """Evaluate the current weights against a random policy."""
    total_reward = 0
    for i in tqdm(range(5), desc="Evaluating best model against 5 dummies", leave=False):  
        reward, _ = play_game(env=env, player1=best_agent.model,
                                    player2=RandomPolicy(env.action_space(env.agents[0]).n),
                                    args=args, eval=True)
        total_reward += reward

        if args.debug:
            print(f"\n\t evaluation number {i}, reward = {reward} ")
    
    return total_reward / 5




def mutate_elites(env, elites, args):
    mutated_elites_list = []
    for i in range(args.population-1):
        if args.debug:
            print(f"\n\t mutate elite {i % args.elites_number}")
        elite = elites[i % args.elites_number]
        mutated_elite = elite.clone(env, args)
        mutated_elite.mutate(args.mutation_power)
        mutated_elites_list.append(mutated_elite)
    return mutated_elites_list


def genetic_algorithm_train(env, agent, args):
    
    """ Evolve the next generation using the Genetic Algorithm. This process
    consists of three steps:
    1. Communicate the elites of the previous generation
    to the workers and let them mutate and evaluate them against individuals from
    the Hall of Fame. To include a form of Elitism, not all elites are mutated.
    2. Communicate the mutated weights and fitnesses back to the trainer and
    determine which of the individuals are the fittest. The fittest individuals
    will form the elites of the next population.
    3. Evaluate the fittest (more rewarded) individual against a random policy and log the results. """

    # Load elites and HoF from disk if they exist
    # Otherwise, just initialize them
    model_dir = create_ga_model_dir(args)
    hof_file = os.path.join(model_dir, "hall_of_fame.pth")
    elite_file = os.path.join(model_dir, "elite_weights.pth")
    plot_file = os.path.join(model_dir, "rewards_plot.png")

    hof = load_hof(hof_file, env, args)
    elites = load_elites(elite_file, env, args)

    rewards_over_generations = []  # Track rewards for each generation


    population = []
    for i in tqdm(range(args.population), desc=f"Creating initial population (n = {args.population})", leave=False):
        agent = create_agent(env, args)
        population.append(agent)

    for gen in tqdm(range(args.generations), desc="Generations"):

        results = []
        population_fitness = []

        # Evaluate mutations vs first hof
        # here mutations happen
        for i in tqdm(range(args.population), desc=f"Population vs HoF elite (k = {args.hof_size})", leave=False):

            individual_reward = 0

            individual = population[i]

            for k in tqdm(range(args.hof_size), desc=f"Individual n.{i} vs HoF elite", leave=False):
                
                hof_elite_member = hof[len(hof)-1-k]
                individual_reward1, hof_reward1 = play_game(env=env, player1=individual.model, player2=hof_elite_member.model, args=args)
                hof_reward2, individual_reward2 = play_game(env=env, player1=hof_elite_member.model, player2=individual.model, args=args)
                individual_reward += individual_reward1 + individual_reward2

                if args.debug:
                    print(f"\n\thof elite {-k} vs individual {i}, individual got reward = {individual_reward1 + individual_reward2}")
            
            individual_fitness = individual_reward / args.hof_size
            
            if args.debug:
                print(f"\nindividual has fitness {individual_fitness}")
                                
            population_fitness.append(individual_fitness)
    
        if args.debug:
            print(f"\npopulation_fitness = {population_fitness}")
        
        ordered_population_fitness = np.argsort(population_fitness)[::-1]
        
        if args.debug:
            print(f"\nordered_population_fitness = {ordered_population_fitness}")

        elite_ids = ordered_population_fitness[:args.elites_number]

        elite_rewards = []

        elites = []

        for idd in elite_ids:
            elite_rewards.append(population_fitness[idd])
            elites.append(population[idd])

            if args.debug:
                print(f"\n elite = {idd}, with reward = {population_fitness[idd]}")


        best_id = elite_ids[0]
        best_agent = population[best_id]
        population = []
        population.append(best_agent)
        
        if args.debug:
            print(f"\nBest of the generation: {best_id}")

        hof.append(best_agent)
        hof.pop(0)                  # for saturation of RAM

        # now we create the new population
        # the best id will be part of it
        # then we mutate the elite of T individuals, obtaining n-1 new individuals
   
        if args.debug:
            print("\nlet's now mutate the elites")
        
        new_mutations = mutate_elites(env, elites, args)

        for new_mutation in new_mutations:
            population.append(new_mutation)


        # Save the HoF and the elites at the end of each generation
        if args.save:
            save_hof(hof, hof_file)
            save_elites(elites, elite_file)

        # Evaluate best mutation vs random agent
        evaluation_reward = 0

        if args.debug:
            print("\nlet's evaluate the best model against a random policy")

        if args.env_mode == "parallel":
            evaluation_reward = evaluate_current_weights_parallel(best_agent.model, env, args=args)
        else:
            evaluation_reward = evaluate_current_weights(best_agent, env, args=args)
        
        if args.debug:
            print(f"\nevaluation reward = {evaluation_reward}")

        # Append evaluation reward for plotting
        rewards_over_generations.append(evaluation_reward)

        # adaptive mutation: TODO: PUT IN THE REPORT THAT THIS WAS NOT IN THE PAPER, IT WAS OUR IDEA
        if args.adaptive:
            args.mutation_power = max(0.01, args.mutation_power * 0.95)  # Reduce over generations

        # Plot and save rewards progression
        average_window = 5  # Define window for moving average (e.g., 5 or 10 generations)
        plot_rewards(rewards_over_generations, plot_file, window=average_window)

    return hof

