import torch
import numpy as np
from deepqn import DeepQN
from agent import Agent
from random import randint
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
#import torchvision.transforms as transforms



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


def create_ga_model_dir(args):
    """Create a directory for saving models based on hyperparameters."""
    dir_name = f"GA_models/gens{args.generations}_pop{args.population}_hof{args.hof_size}_game{args.atari_game}_mut{args.mutation_power}_adaptive{args.adaptive}_tslimit{args.max_timesteps_per_episode}"
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
        hof = [Agent(
                input_channels=args.input_channels,
                n_actions=env.action_space(env.agents[0]).n,
                precision=args.precision,
            )
            for _ in range(args.hof_size)
        ]
    return hof

def load_elites(file_path, env=None, args=None):
    """ Load the elites from disk. """
    elites_agents = []
    if os.path.exists(file_path):
        elites_agents = torch.load(file_path)
        print(f"Elites loaded from {file_path}")
    else:
        print(f"No elites found at {file_path}. Returning initial list of weights.")
        elites_agents = [
            Agent(
                input_channels=args.input_channels,
                n_actions=env.action_space(env.agents[0]).n,
                precision=args.precision,
            )
            for _ in range(args.elites_number)
        ]

        for agent in elites_agents:
            if args.precision == "float16":
                agent.model.half()
            else:
                agent.model.float()

    return elites_agents


class RandomPolicy(object):
    """ Policy that samples a random move from the number of actions that
    are available."""

    def __init__(self, number_actions):
        self.number_actions = number_actions

    def determine_action(self, input):
        return randint(0, self.number_actions - 1)

"""
def evaluate_current_weights_parallel(best_mutation_weights, env, args):
    # Initialize agents
    agents = {}
    for i, agent_id in enumerate(env.agents):
        if i == 0:  # The elite agent
            agents[agent_id] = Agent(args.input_channels, env.action_space(agent_id).n, precision=args.precision)
            agents[agent_id].set_weights(best_mutation_weights)
        else:  # Random policy agent(s)
            agents[agent_id] = RandomPolicy(env.action_space(agent_id).n)

    # Play a parallel game
    rewards, timesteps = play_game_parallel(env, agents, args, eval=True)

    # Return the total reward of the elite agent (assumed to be agent 0)
    elite_id = env.agents[0]
    return {
        'total_reward': rewards[elite_id],
        'timesteps_total': timesteps,
    }
"""


def evaluate_current_weights(best_agent, env, args):
    """Evaluate the current weights against a random policy."""
    total_reward = 0
    for i in tqdm(range(5), desc="Evaluating best model against 5 dummies", leave=False):  
        reward, _, ts = play_game(env=env, player1=best_agent.model,
                                    player2=RandomPolicy(env.action_space(env.agents[0]).n),
                                    args=args, eval=True)
        total_reward += reward

        if args.debug:
            print(f"\n\t evaluation number {i}, reward = {reward} ")
    
    return total_reward / 5



def play_game_parallel(env, agents, args, eval=False):
    # TODO: add the possibility of having no timesteps_limit
    """Play a game using the weights of all agents simultaneously in the parallel environment."""
    env.reset()
    rewards = {agent: 0 for agent in env.agents}
    timesteps = 0
    timesteps_limit = args.max_evaluation_steps if eval else args.max_timesteps_per_episode
    done = False

    while not done and timesteps < timesteps_limit:
        # Gather actions from all agents
        actions = {}
        for agent in env.agents:
            obs = env.observe(agent)
            dtype = torch.float16 if args.precision == "float16" else torch.float32
            obs = torch.from_numpy(obs).to(dtype).unsqueeze(0).permute(0, 3, 1, 2) # Convert to [N, C, H, W]
            actions[agent] = agents[agent].determine_action(obs)
        
        # Step environment with all actions
        observations, step_rewards, dones, infos = env.step(actions)
        
        # Update rewards and check if all agents are done
        for agent, reward in step_rewards.items():
            rewards[agent] += reward
        timesteps += 1
        done = all(dones.values())

    return rewards, timesteps


def play_game(env, player1, player2, args, eval=False):
    """Play a game using the weights of two players in the PettingZoo environment."""
    env.reset()
    rewards = {"first_0": 0, "second_0": 0}
    timesteps = 0
    timesteps_limit = args.max_evaluation_steps if eval else args.max_timesteps_per_episode
    done = False

    for agent in env.agent_iter():
        obs = env.observe(agent)
        obs = torch.from_numpy(obs) # Ensure it's of the correct dtype
        if args.precision == "float16":
            obs = obs.to(torch.float16)
        else:
            obs = obs.to(torch.float32)
        obs = obs.permute(2, 0, 1)   # N x W x H (6 x 84 x 84)
        obs = obs.unsqueeze(0) # 1 x N x W x H
        if agent == "first_0":   
            action = player1.determine_action(obs)
        elif agent == "second_0":
            action = player2.determine_action(obs)  
        else:
            raise ValueError(f"Unknown Agent during play_game: {agent}")
        env.step(action)    
        _, reward, termination, truncation, _ = env.last()

        """
        if args.debug:
            if reward != 0:
                print(f"\nagent = {agent}, reward {reward}, termination = {termination}, truncation = {truncation}")
        """

        rewards[agent] += reward
        timesteps += 1

    
        if timesteps_limit is not None and timesteps >= timesteps_limit:
            
            """
            if args.debug:
                print(f"timesteps limit imposed reached!")
            """

            break

        if termination or truncation:
            # If the environment has only two agents (like Pong), 
            # breaking immediately on termination or truncation is generally okay because once one agent is done, 
            # the other agent’s turn will immediately follow (in the round-robin order). After both agents have 
            # been processed, the episode ends naturally. For more complex environments with multiple agents, 
            # handling termination per agent ensures consistency.
            
            """
            if args.debug:
                print(f"termination or truncation is true")
            """

            break

    return rewards["first_0"], rewards["second_0"], timesteps


def evaluate_mutations_parallel(env, elite_weights, opponent_weights, args, mutate_opponent=True):
    """Evaluate mutations in parallel mode."""
    # Initialize agents
    agents = {}
    for i, agent_id in enumerate(env.agents):
        if i % 2 == 0:
            agents[agent_id] = Agent(args.input_channels, env.action_space(agent_id).n, precision=args.precision)
            agents[agent_id].set_weights(elite_weights)
        else:
            agents[agent_id] = Agent(args.input_channels, env.action_space(agent_id).n, precision=args.precision)
            agents[agent_id].set_weights(opponent_weights)
            if mutate_opponent:
                agents[agent_id].mutate(args.mutation_power)

    # Play two games: agent1 vs agent2 and agent2 vs agent1
    rewards1, ts1 = play_game_parallel(env, agents, args)
    rewards2, ts2 = play_game_parallel(env, agents, args)

    # Compute total rewards for each agent
    total_rewards = {agent: rewards1[agent] + rewards2[agent] for agent in rewards1.keys()}

    return {
        'opponent_weights': agents[env.agents[1]].get_weights(),  # Return the mutated opponent weights
        'score_vs_elite': total_rewards[env.agents[0]],  # Elite's total score
        'timesteps_total': ts1 + ts2,
    }


def mutate_elites(elites, args):
    mutated_elites_list = []
    for i in range(args.population-1):
        if args.debug:
            print(f"\n\t mutate elite {i % args.elites_number}")
        elite = elites[i % args.elites_number]
        mutated_elite = elite.clone(args)
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
        agent = Agent(input_channels=args.input_channels, n_actions=env.action_space(env.agents[0]).n, precision=args.precision)
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
                individual_reward1, hof_reward1, ts1 = play_game(env=env, player1=individual.model, player2=hof_elite_member.model, args=args)
                hof_reward2, individual_reward2, ts2 = play_game(env=env, player1=hof_elite_member.model, player2=individual.model, args=args)
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
        
        new_mutations = mutate_elites(elites, args)

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


