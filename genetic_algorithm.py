import torch
import numpy as np
from utils.game_logic_functions import create_agent, play_game, diversity_penalty
from utils.utils_pth_and_plots import plot_experiment_metrics, save_model
from utils.game_logic_functions import initialize_env
from utils.utils_policies import RandomPolicy
from agent import Agent
from tqdm import tqdm
import os


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


    return total_reward_agent_0 / 10, total_reward_agent_1 / 10, total_reward_adversary / 10


def mutate_elites(env, elites, args, role):
    mutated_elites_list = []
    for i in range(args.population-1):
        
        if role == 'agent_0':
            mutation_power = args.mutation_power_agent_0
        elif role == 'agent_1':
            mutation_power = args.mutation_power_agent_1
        else:
            mutation_power = args.mutation_power_adversary
            
        elite = elites[i % args.elites_number]
        mutated_elite = elite.clone(env, args, role)
        mutated_elite.mutate(mutation_power)
        mutated_elites_list.append(mutated_elite)   
         
    return mutated_elites_list


def genetic_algorithm_train(env, agent, args, output_dir):
 
    hof_file_agent_0 = os.path.join(output_dir, "hall_of_fame.pth")
    elite_file_agent_0 = os.path.join(output_dir, "elite_weights.pth")
    results_plot_file_agent_0 = os.path.join(output_dir, "results_plot.png")
    hof_file_agent_1 = os.path.join(output_dir, "hall_of_fame.pth")
    elite_file_agent_1 = os.path.join(output_dir, "elite_weights.pth")
    results_plot_file_agent_1 = os.path.join(output_dir, "results_plot.png")
    hof_file_adversary = os.path.join(output_dir, "hall_of_fame.pth")
    elite_file_adversary = os.path.join(output_dir, "elite_weights.pth")
    results_plot_file_adversary = os.path.join(output_dir, "results_plot.png")

    hof_agent_1 = [create_agent(env, args, "agent_1") for _ in range(args.hof_size)]
    hof_agent_0 = [create_agent(env, args,  "agent_0") for _ in range(args.hof_size)]
    hof_adversary = [create_agent(env, args, "adversary_0") for _ in range(args.hof_size)]
    elites_agent_0 = [create_agent(env, args, "agent_1") for _ in range(args.hof_size)]
    elites_agent_1 = [create_agent(env, args, "agent_0") for _ in range(args.hof_size)]
    elites_adversary = [create_agent(env, args, "adversary_0") for _ in range(args.hof_size)]

    # TODO, put precision on the agents            

    total_params = sum(p.numel() for p in hof_agent_1[0].model.parameters()) 
    print(f'\nNumber of parameters for agent_0 network: {total_params}')
    total_params = sum(p.numel() for p in hof_agent_0[0].model.parameters()) 
    print(f'\nNumber of parameters for agent_1 network: {total_params}')
    total_params = sum(p.numel() for p in hof_adversary[0].model.parameters()) 
    print(f'\nNumber of parameters for adversary network: {total_params}')

    rewards_over_generations_agent_0 = []
    rewards_over_generations_agent_1 = []
    rewards_over_generations_adversary = []
    
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

    population_agent_0 = []
    population_agent_1 = []
    population_adversary = []
    for i in tqdm(range(args.population), desc=f"Creating initial population (n = {args.population})", leave=False):

        agent_0 = create_agent(env, args, "agent_0")
        population_agent_0.append(agent_0)
        agent_1 = create_agent(env, args, "agent_1")
        population_agent_1.append(agent_1)
        adversary = create_agent(env, args, "adversary_0")
        population_adversary.append(adversary)

    for gen in tqdm(range(args.generations), desc="Generations"):

        population_fitness_agent_0 = []
        population_fitness_agent_1 = []
        population_fitness_adversary = []

        for i in tqdm(range(args.population), desc=f"Agent 0 Population vs HoF elite (k = {args.hof_size})", leave=False):

            agent_0_reward = 0

            individual_agent_0 = population_agent_0[i]
            
            population_weights_agent_0 = [population_agent_0[i].model.get_weights_ES() for i in range(len(population_agent_0))]
            
            diversity_agent_0 = diversity_penalty(individual_weights=agent_0.model.get_weights_ES(), population_weights=population_weights_agent_0, args=args)
            

            for k in tqdm(range(args.hof_size), desc=f"Individual n.{i} vs HoF elite", leave=False):
                
                hof_elite_member_agent_1 = hof_agent_1[len(hof_agent_1)-1-k]
                hof_elite_member_adversary = hof_adversary[len(hof_adversary)-1-k]
                agent_0_reward, agent_1_reward, adversary_reward = play_game(env=env, player1=individual_agent_0.model, 
                                                                                    player2=hof_elite_member_agent_1.model,
                                                                                    adversary=hof_elite_member_adversary.model, args=args)
                
            total_agent_0_reward = agent_0_reward / args.hof_size
            
            total_agent_0_fitness = total_agent_0_reward / (1 + diversity_agent_0)
            
            if args.debug:
                print(f"\nindividual has fitness {total_agent_0_fitness}")
                                
            population_fitness_agent_0.append(total_agent_0_fitness)
    
        if args.debug:
            print(f"\npopulation_fitness = {population_fitness_agent_0}")


        for i in tqdm(range(args.population), desc=f"Agent 1 Population vs HoF elite (k = {args.hof_size})", leave=False):

            agent_1_reward = 0

            individual_agent_1 = population_agent_1[i]
            
            population_weights_agent_1 = [population_agent_1[i].model.get_weights_ES() for i in range(len(population_agent_1))]
            
            diversity_agent_1 = diversity_penalty(individual_weights=agent_1.model.get_weights_ES(), population_weights=population_weights_agent_1, args=args)
            

            for k in tqdm(range(args.hof_size), desc=f"Individual n.{i} vs HoF elite", leave=False):
                
                hof_elite_member_agent_0 = hof_agent_0[len(hof_agent_0)-1-k]
                hof_elite_member_adversary = hof_adversary[len(hof_adversary)-1-k]
                agent_0_reward, agent_1_reward, adversary_reward = play_game(env=env, player1=hof_elite_member_agent_0.model, 
                                                                                    player2=individual_agent_1.model,
                                                                                    adversary=hof_elite_member_adversary.model, args=args)
                
            total_agent_1_reward = agent_1_reward / args.hof_size
            
            total_agent_1_fitness = total_agent_1_reward / (1 + diversity_agent_1)
            
            
            if args.debug:
                print(f"\nindividual has fitness {total_agent_1_fitness}")
                                
            population_fitness_agent_1.append(total_agent_1_fitness)
    
        if args.debug:
            print(f"\npopulation_fitness = {population_fitness_agent_1}")



        for i in tqdm(range(args.population), desc=f"Adversary Population vs HoF elite (k = {args.hof_size})", leave=False):

            adversary_reward = 0

            individual_adversary = population_adversary[i]
            
            population_weights_adversary = [population_adversary[i].model.get_weights_ES() for i in range(len(population_adversary))]
            
            diversity_adversary = diversity_penalty(individual_weights=adversary.model.get_weights_ES(), population_weights=population_weights_adversary, args=args)

            for k in tqdm(range(args.hof_size), desc=f"Individual n.{i} vs HoF elite", leave=False):
                
                hof_elite_member_agent_0 = hof_agent_0[len(hof_agent_0)-1-k]
                hof_elite_member_agent_1 = hof_agent_0[len(hof_agent_1)-1-k]
                agent_0_reward, agent_1_reward, adversary_reward = play_game(env=env, player1=hof_elite_member_agent_0.model, 
                                                                                    player2=hof_elite_member_agent_1.model,
                                                                                    adversary=individual_adversary.model, args=args)
                
            total_adversary_reward = adversary_reward / args.hof_size
            
            total_adversarry_fitness = total_adversary_reward / (1 + diversity_adversary)
            
            
            if args.debug:
                print(f"\nindividual has fitness {total_adversarry_fitness}")
                                
            population_fitness_adversary.append(total_adversarry_fitness)
    
        if args.debug:
            print(f"\npopulation_fitness = {total_adversarry_fitness}")
        

        ordered_population_fitness_agent_0 = np.argsort(population_fitness_agent_0)[::-1]
        ordered_population_fitness_agent_1 = np.argsort(population_fitness_agent_1)[::-1]
        ordered_population_fitness_adversary = np.argsort(population_fitness_adversary)[::-1]
        
        if args.debug:
            print(f"\nordered_population_fitness = {population_fitness_agent_0}")
            print(f"\nordered_population_fitness = {population_fitness_agent_1}")
            print(f"\nordered_population_fitness = {population_fitness_adversary}")

        elite_ids_agent_0 = ordered_population_fitness_agent_0[:args.elites_number]
        elite_ids_agent_1 = ordered_population_fitness_agent_1[:args.elites_number]
        elite_ids_adversary = ordered_population_fitness_adversary[:args.elites_number]

        elite_rewards_agent_0 = []
        elite_rewards_agent_1 = []
        elite_rewards_adversary = []

        elites_agent_0 = []
        elites_agent_1 = []
        elites_adversary = []

        for idd in elite_ids_agent_0:
            elite_rewards_agent_0.append(population_fitness_agent_0[idd])
            elites_agent_0.append(population_agent_0[idd])
        for idd in elite_ids_agent_1:
            elite_rewards_agent_1.append(population_fitness_agent_1[idd])
            elites_agent_1.append(population_agent_1[idd])
        for idd in elite_ids_adversary:
            elite_rewards_adversary.append(population_fitness_adversary[idd])
            elites_adversary.append(population_adversary[idd])


        best_id_agent_0 = elite_ids_agent_0[0]
        best_agent_0 = population_agent_0[best_id_agent_0]
        population_agent_0 = []
        population_agent_0.append(best_agent_0)
        
        best_id_agent_1 = elite_ids_agent_1[0]
        best_agent_1 = population_agent_1[best_id_agent_1]
        population_agent_1 = []
        population_agent_1.append(best_agent_1)
        
        best_id_adversary = elite_ids_adversary[0]
        best_adversary = population_adversary[best_id_adversary]
        population_adversary = []
        population_adversary.append(best_adversary)
        
        hof_agent_0.append(best_agent_0)
        hof_agent_0.pop(0)                  
        hof_agent_1.append(best_agent_1)
        hof_agent_1.pop(0)                  
        hof_adversary.append(best_adversary)
        hof_adversary.pop(0)                  

        # now we create the new population
        # the best id will be part of it
        # then we mutate the elite of T individuals, obtaining n-1 new individuals
   
        new_mutations_agent_0 = mutate_elites(env, elites_agent_0, args, "agent_0")
        new_mutations_agent_1 = mutate_elites(env, elites_agent_1, args, "agent_1")
        new_mutations_adversary = mutate_elites(env, elites_adversary, args, "adversary_0")

        for new_mutation_agent_0 in new_mutations_agent_0:
            population_agent_0.append(new_mutation_agent_0)
        for new_mutation_agent_1 in new_mutations_agent_1:
            population_agent_1.append(new_mutation_agent_1)
        for new_mutation_adversary in new_mutations_adversary:
            population_adversary.append(new_mutation_adversary)

        # Save the HoF and the elites at the end of each generation
        if args.save:
            save_model(hof_agent_0, hof_file_agent_0)
            save_model(elites_agent_0, elite_file_agent_0)
            save_model(hof_agent_0, hof_file_agent_0)
            save_model(elites_agent_0, elite_file_agent_0)
            save_model(hof_agent_0, hof_file_agent_0)
            save_model(elites_agent_0, elite_file_agent_0)

        evaluation_reward_agent_0, evaluation_reward_agent_1, evaluation_reward_adversary = evaluate_current_weights(best_agent_0, best_agent_1, best_adversary, env, args=args)

        # Append evaluation reward for plotting
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


