import torch
import numpy as np
from deepqn import DeepQN
from agent import Agent
from random import randint
from tqdm import tqdm
import os
import matplotlib.pyplot as plt



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


def load_hof(file_path, args=None, elites=None):
    """ Load the Hall of Fame (HoF) from disk. """
    if os.path.exists(file_path):
        hof = torch.load(file_path)
        print(f"Hall of Fame loaded from {file_path}")
    else:
        print(f"No HoF file found at {file_path}. Returning initial list of weights.")
        hof = [elites[i].get_weights() for i in range(args.elites_number)]
    return hof

def load_elites(file_path, args=None, env=None):
    """ Load the elites from disk. """
    if os.path.exists(file_path):
        elites = torch.load(file_path)
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

        elites = [agent.model for agent in elites_agents]
    
    return elites


class RandomPolicy(object):
    """ Policy that samples a random move from the number of actions that
    are available."""

    def __init__(self, number_actions):
        self.number_actions = number_actions

    def determine_action(self, input):
        return randint(0, self.number_actions - 1)


def evaluate_current_weights_parallel(best_mutation_weights, env, args):
    """Evaluate weights by playing against a random policy in parallel mode."""
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


def evaluate_current_weights(best_mutation_weights, env, args):
    """ Evaluate weights by playing against a random policy. """
    elite = Agent(3, env.action_space(env.agents[0]).n, precision=args.precision)
    elite.set_weights(best_mutation_weights)
    reward, _, ts = play_game(env=env, player1=elite.model,
                                    player2=RandomPolicy(env.action_space(env.agents[0]).n), 
                                    args=args, eval=True)
    return {
        'total_reward': reward,
        'timesteps_total': ts,
    }


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
        obs = torch.from_numpy(obs).float()  # Ensure it's of the correct dtype
        obs = torch.unsqueeze(obs, dim=0)  # Shape becomes [1, 210, 160, 3] 
        obs = obs.permute(0, 3, 1, 2)  # Convert from [N, H, W, C] to [N, C, H, W]

        #print(f"agent = {agent}")
        if agent == "first_0":   
            action = player1.determine_action(obs)
            #print(f"action chosen = {action}")
        elif agent == "second_0":
            action = player2.determine_action(obs)
        else:
            raise ValueError(f"Unknown Agent during play_game: {agent}")
            #action = env.action_space(agent).sample()  # Random fallback action

        env.step(action)    
        _, reward, termination, truncation, _ = env.last()
        
        if args.debug:
            if reward != 0:
                print(f"reward {agent} = {reward}, termination = {termination}, truncation = {truncation}")
                print(f"Action chosen by {agent}: {action}")

         
        rewards[agent] += reward
        timesteps += 1

    
        if timesteps_limit is not None and timesteps >= timesteps_limit:
            break

        if termination or truncation:
            # If the environment has only two agents (like Pong), 
            # breaking immediately on termination or truncation is generally okay because once one agent is done, 
            # the other agent’s turn will immediately follow (in the round-robin order). After both agents have 
            # been processed, the episode ends naturally. For more complex environments with multiple agents, 
            # handling termination per agent ensures consistency.
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


def evaluate_mutations(env, elite_weights, opponent_weights, args, mutate_opponent=True):
    """ Mutate (sometimes) the inputted weights and evaluate its performance against the inputted opponent. """
    elite = Agent(args.input_channels, n_actions=env.action_space(env.agents[0]).n, precision=args.precision) # first_0 since the actions set is equal for both agents
    elite.model.set_weights(elite_weights)
    opponent = Agent(args.input_channels, n_actions=env.action_space(env.agents[0]).n, precision=args.precision)
    opponent.model.set_weights(opponent_weights)

    if mutate_opponent:
        opponent.mutate(args.mutation_power)

    elite_reward1, opponent_reward1, ts1 = play_game(env=env, player1=elite.model, player2=opponent.model, args=args)
    opponent_reward2, elite_reward2, ts2 = play_game(env=env, player1=opponent.model, player2=elite.model, args=args)
    
    total_elite = elite_reward1 + elite_reward2
    total_opponent = opponent_reward1 + opponent_reward2
    return {
        'opponent_weights': opponent.model.get_weights(),
        'score_vs_elite': total_opponent,
        'timesteps_total': ts1 + ts2,
    }



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

    elites = load_elites(elite_file, args, env)
    hof = load_hof(hof_file, args, elites)

    rewards_over_generations = []  # Track rewards for each generation

    for gen in tqdm(range(args.generations), desc="Generations"):

        results = []
        
        # Evaluate mutations vs first hof
        # here mutations happen
        for i in tqdm(range(args.population), desc=f"Evaluating Elite vs Youngest HoF (Gen {gen})", leave=False):
            elite_id = i % args.elites_number
            should_mutate = (i > args.elites_number)
            if args.env_mode == "parallel":
                results.append(evaluate_mutations_parallel(env, hof[-1], elites[elite_id].get_weights(), args, mutate_opponent=should_mutate))
            else:
                results.append(evaluate_mutations(env, hof[-1], elites[elite_id].get_weights(), args, mutate_opponent=should_mutate))
            
            if args.debug:
                print(f"\nelite {elite_id} vs Best of the HoF")
                print(f"\n\t reward: {results[-1]['score_vs_elite']}")
        

        # Evaluate vs other hof
        for j in tqdm(range(len(hof) - 1), desc=f"Evaluating Population vs the rest of the HoF (Gen {gen})", leave=False):
            with tqdm(range(args.population), desc=f"Individuals vs HoF n.{len(hof)-2-j} (Gen {gen})", leave=False) as inner_pbar:
                for i in inner_pbar:

                    if args.env_mode == "parallel":
                        results.append(evaluate_mutations_parallel(env, elite_weights=hof[-2 - j], opponent_weights=results[i]['opponent_weights'], args=args, mutate_opponent=False))
                    else:
                        results.append(evaluate_mutations(env, elite_weights=hof[-2 - j], opponent_weights=results[i]['opponent_weights'], args=args, mutate_opponent=False))
            
                    
                    if args.debug:
                        inner_pbar.set_postfix(individual=i, reward=results[-1]['score_vs_elite'])
                        print(f"individual {i} vs HoF n.{len(hof)-2-j}")
                        print(f"\t reward: {results[-1]['score_vs_elite']}")

        rewards = []

        #print(len(results))
        for i in range(args.population):
            total_reward = 0
            for j in range(len(hof)):
                reward_index = args.population * j + i
                total_reward += results[reward_index]['score_vs_elite']
            rewards.append(total_reward)

        best_mutation_id = np.argmax(rewards)
        best_mutation_weights = results[best_mutation_id]['opponent_weights']



        if args.debug:
            print(f"Best mutation: {best_mutation_id} with reward {np.max(rewards)}")


        #self.try_save_winner(best_mutation_weights)
        if len(hof) < args.hof_size:
            hof.append(best_mutation_weights)
        else:
            hof.pop(0)
            hof.append(best_mutation_weights)



        # Elitism retains the best agents from each generation, ensuring the algorithm doesn't lose progress:
        new_elite_ids = np.argsort(rewards)[-args.elites_number:]
        #print(f"TOP mutations: {new_elite_ids}")
        for i, elite in enumerate(elites):
            mutation_id = new_elite_ids[i]
            elite.set_weights(results[mutation_id]['opponent_weights'])


        # Save the HoF and the elites at the end of each generation
        if args.save:
            save_hof(hof, hof_file)
            save_elites(elites, elite_file)

        # Evaluate best mutation vs random agent
        evaluate_results = 0

        if args.env_mode == "parallel":
            evaluate_results = evaluate_current_weights_parallel(best_mutation_weights, env, args=args)
        else:
            evaluate_results = evaluate_current_weights(best_mutation_weights, env, args=args)
        
        evaluate_rewards = evaluate_results['total_reward']
        
        if args.debug:
            print(f"\ngen's best evaluation reward = {evaluate_rewards}")

        #evaluate_videos = [result['video'] for result in evaluate_results]

        #increment_metrics(results)

        # Append evaluation reward for plotting
        rewards_over_generations.append(evaluate_rewards)

        # adaptive mutation: TODO: PUT IN THE REPORT THAT THIS WAS NOT IN THE PAPER, IT WAS OUR IDEA
        if args.adaptive:
            args.mutation_power = max(0.01, args.mutation_power * 0.95)  # Reduce over generations

        # Plot and save rewards progression
        average_window = 5  # Define window for moving average (e.g., 5 or 10 generations)
        plot_rewards(rewards_over_generations, plot_file, window=average_window)

    return hof


