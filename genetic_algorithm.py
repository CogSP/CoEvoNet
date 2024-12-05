import torch
import numpy as np
from deepqn import DeepQN
from agent import Agent
from random import randint
from tqdm import tqdm


class RandomPolicy(object):
    """ Policy that samples a random move from the number of actions that
    are available."""

    def __init__(self, number_actions):
        self.number_actions = number_actions

    def determine_action(self, input):
        return randint(0, self.number_actions - 1)
    

def evaluate_current_weights(best_mutation_weights, env, args):
        """ Evaluate weights by playing against a random policy. """
        elite = Agent(3, env.action_space(env.agents[0]).n)
        elite.set_weights(best_mutation_weights)
        reward, _, ts = play_game(env=env, player1=elite.model,
                                       player2=RandomPolicy(env.action_space(env.agents[0]).n), 
                                       args=args, eval=True)
        return {
            'total_reward': reward,
            'timesteps_total': ts,
        }


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
            action = env.action_space(agent).sample()  # Random fallback action

        env.step(action)
        _, reward, _, done, _ = env.last()
      
        rewards[agent] += reward
        timesteps += 1
        #print(dir(env))
    
        if done or timesteps >= timesteps_limit:

            if args.debug:
                if done:
                    print("\nsomeone won\n")
                else:
                    print("\ntime's out\n")
            break

    return rewards["first_0"], rewards["second_0"], timesteps


def evaluate_mutations(env, elite_weights, opponent_weights, args, mutate_opponent=True):
    """ Mutate (sometimes) the inputted weights and evaluate its performance against the inputted opponent. """
    elite = Agent(args.input_channels, n_actions=env.action_space(env.agents[0]).n) # first_0 since the actions set is equal for both agents
    elite.model.set_weights(elite_weights)
    opponent = Agent(args.input_channels, n_actions=env.action_space(env.agents[0]).n)
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
    3. Evaluate the fittest (more rewarder) individual against a random policy and log the results. """

    elites = [DeepQN(input_channels=args.input_channels, n_actions=env.action_space(agent).n) for _ in range(args.elites_number)]

    # TODO: do things with VBN

    hof = [elites[i].get_weights() for i in range(args.elites_number)]

    for gen in tqdm(range(args.generations), desc="Generations"):


        results = []
        
        # Evaluate mutations vs first hof
        # here mutations happen
        for i in tqdm(range(args.population), desc=f"Evaluating Elitest vs Youngest HoF (Gen {gen})", leave=False):
            elite_id = i % args.elites_number
            should_mutate = (i > args.elites_number)
            results += [evaluate_mutations(env=env, elite_weights=hof[-1], opponent_weights=elites[elite_id].get_weights(), args=args, mutate_opponent=should_mutate)]
            
            if args.debug:
                print(f"\nelite {elite_id} vs Best of the HoF")
                print(f"\n\t reward: {results[-1]['score_vs_elite']}")


        # Evaluate vs other hof
        for j in tqdm(range(len(hof) - 1), desc=f"Evaluating Population vs the rest of the HoF (Gen {gen})", leave=False):
            with tqdm(range(args.population), desc=f"Individuals vs HoF n.{len(hof)-2-j} (Gen {gen})", leave=False) as inner_pbar:
                for i in inner_pbar:
                    results += [evaluate_mutations(env, elite_weights=hof[-2 - j], opponent_weights=results[i]['opponent_weights'], args=args, mutate_opponent=False)]
                    
                    if args.debug:
                        inner_pbar.set_postfix(individual=i, reward=results[-1]['score_vs_elite'])
                        print(f"individual {i} vs HoF n.{len(hof)-2-j}")
                        print(f"\t reward: {results[-1]['score_vs_elite']}")

        rewards = []
        #print(len(results))
        for i in range(args.population):
            total_reward = 0
            for j in range(args.elites_number):
                reward_index = args.population * j + i
                total_reward += results[reward_index]['score_vs_elite']
            rewards.append(total_reward)

        best_mutation_id = np.argmax(rewards)
        best_mutation_weights = results[best_mutation_id]['opponent_weights']
        
        if args.debug:
            print(f"Best mutation: {best_mutation_id} with reward {np.max(rewards)}")

        #self.try_save_winner(best_mutation_weights)
        hof.append(best_mutation_weights)


        #Elitism retains the best agents from each generation, ensuring the algorithm doesn't lose progress:
        new_elite_ids = np.argsort(rewards)[-args.elites_number:]
        #print(f"TOP mutations: {new_elite_ids}")
        for i, elite in enumerate(elites):
            mutation_id = new_elite_ids[i]
            elite.set_weights(results[mutation_id]['opponent_weights'])

        # Evaluate best mutation vs random agent
        evaluate_results = evaluate_current_weights(best_mutation_weights, env, args=args)
        evaluate_rewards = evaluate_results['total_reward']
        
        if args.debug:
            print(f"\ngen's best evaluation reward = {evaluate_rewards}")

        train_rewards = [result['score_vs_elite'] for result in results]

        #evaluate_videos = [result['video'] for result in evaluate_results]

        #increment_metrics(results)

        # adaptive mutation: TODO: PUT IN THE REPORT THAT THIS WAS NOT IN THE PAPER, IT WAS OUR IDEA
        args.mutation_power = max(0.01, args.mutation_power * 0.95)  # Reduce over generations

    return 


