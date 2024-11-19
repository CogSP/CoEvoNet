import torch
import numpy as np
from deepqn import DeepQN

ELITES_NUMBER = 1
POPULATION_SIZE = 1
MUTATION_POWER = 0.05
LEARNING_RATE = 0.1
TIMESTEPS_TOTAL = 0
EPISODES_TOTAL = 0
GENERATION = 0
MAX_EVALUATION_STEPS = 500
INPUT_CHANNEL = 3
N_ACTIONS= 5


def evaluate_current_weights(best_mutation_weights):
    evaluate_jobs = []
    for i in range(self.config['evaluation_games']):
        worker_id = i % self.config['num_workers']
        evaluate_jobs += [self._workers[worker_id].evaluate.remote(
            best_mutation_weights)]
    evaluate_results = ray.get(evaluate_jobs)
    return evaluate_results

def increment_metrics(self, results):
    """ Increment the total timesteps, episodes and generations. """
    TIMESTEPS_TOTAL += sum([result['timesteps_total'] for result in results])
    EPISODES_TOTAL += len(results)
    GENERATION += 1


# Fitness function to evaluate an agent
def run_episode(env, agent):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_values = agent.model(state_tensor)
        action = torch.argmax(action_values, dim=1).item()  # Select action
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    return total_reward



def play_game(env, player1, player2, eval=False):
    """Play a game using the weights of two players in the PettingZoo environment."""
    env.reset()
    rewards = {player1: 0, player2: 0}
    timesteps = 0

    for agent in env.agent_iter():
        obs = env.observe(agent)
        obs = torch.from_numpy(obs).float()  # Ensure it's of the correct dtype
        obs = torch.unsqueeze(obs, dim=0)  # Shape becomes [1, 210, 160, 3] 
        obs = obs.permute(0, 3, 1, 2)  # Convert from [N, H, W, C] to [N, C, H, W]

        print(f"agent = {agent}")
        if agent == "first_0":   
            action = player1.determine_action(obs)
            print(f"action chosen = {action}")
        elif agent == "agent_1":
            action = player2.determine_action(obs)
        else:
            action = env.action_space(agent).sample()  # Random fallback action

        env.step(action)
        _, reward, _, _, _ = env.last()

        rewards[agent] += reward
        timesteps += 1
        if all(env.terminated.values()) or timesteps >= MAX_EVALUATION_STEPS:
            break

    return rewards["agent_0"], rewards["agent_1"], timesteps


def evaluate_mutations(env, elite_weights, opponent_weights, mutate_opponent=True):
    """ Mutate the inputted weights and evaluate its performance against the inputted opponent. """
    elite = DeepQN(input_channels=INPUT_CHANNEL, n_actions=N_ACTIONS)
    elite.set_weights(elite_weights)
    opponent = DeepQN(input_channels=INPUT_CHANNEL, n_actions=N_ACTIONS)
    opponent.set_weights(opponent_weights)

    if mutate_opponent:
        opponent.mutate(MUTATION_POWER)
    elite_reward1, opponent_reward1, ts1 = play_game(env, elite, opponent)
    opponent_reward2, elite_reward2, ts2 = play_game(env, opponent, elite)
    total_elite = elite_reward1 + elite_reward2
    total_opponent = opponent_reward1 + opponent_reward2
    return {
        'opponent_weights': opponent.get_weights(),
        'score_vs_elite': total_opponent,
        'timesteps_total': ts1 + ts2,
    }



def genetic_algorithm_train(env, agent, hyperparams, MAX_GENERATIONS):
    
    """ Evolve the next generation using the Genetic Algorithm. This process
    consists of three steps:
    1. Communicate the elites of the previous generation
    to the workers and let them mutate and evaluate them against individuals from
    the Hall of Fame. To include a form of Elitism, not all elites are mutated.
    2. Communicate the mutated weights and fitnesses back to the trainer and
    determine which of the individuals are the fittest. The fittest individuals
    will form the elites of the next population.
    3. Evaluate the fittest
    individual against a random policy and log the results. """


    elites = [DeepQN(input_channels=INPUT_CHANNEL, n_actions=N_ACTIONS) for _ in range(ELITES_NUMBER)]

    # TODO: do things with VBN

    hof = [elites[i].get_weights() for i in range(ELITES_NUMBER)]

    for gen in range(MAX_GENERATIONS):

        results = []
        
        # Evaluate mutations vs first hof
        # here mutations happen
        for i in range(POPULATION_SIZE):
            elite_id = i % ELITES_NUMBER
            should_mutate = (i > ELITES_NUMBER) # elitarism (?)
            results += [evaluate_mutations(env=env, elite_weights=hof[-1], opponent_weights=elites[elite_id].get_weights(), mutate_opponent=should_mutate)]


        # Evaluate vs other hof
        for j in range(len(hof) - 1):
            for i in range(POPULATION_SIZE):
                results += [evaluate_mutations.remote(env, elite_weights=hof[-2 - j], opponent_weights=results[i]['opponent_weights'], mutate_opponent=False)]

        rewards = []
        print(len(results))
        for i in range(POPULATION_SIZE):
            total_reward = 0
            for j in range(ELITES_NUMBER):
                reward_index = POPULATION_SIZE * j + i
                total_reward += results[reward_index]['score_vs_elite']
            rewards.append(total_reward)

        best_mutation_id = np.argmax(rewards)
        best_mutation_weights = results[best_mutation_id]['opponent_weights']
        print(f"Best mutation: {best_mutation_id} with reward {np.max(rewards)}")

        #self.try_save_winner(best_mutation_weights)
        hof.append(best_mutation_weights)

        new_elite_ids = np.argsort(rewards)[-ELITES_NUMBER:]
        print(f"TOP mutations: {new_elite_ids}")
        for i, elite in enumerate(elites):
            mutation_id = new_elite_ids[i]
            elite.set_weights(results[mutation_id]['opponent_weights'])

        # Evaluate best mutation vs random agent
        evaluate_results = evaluate_current_weights(best_mutation_weights)
        evaluate_rewards = [result['total_reward'] for result in evaluate_results]

        train_rewards = [result['score_vs_elite'] for result in results]
        evaluate_videos = [result['video'] for result in evaluate_results]

        increment_metrics(results)

        summary = dict(
            timesteps_total=TIMESTEPS_TOTAL,
            episodes_total=EPISODES_TOTAL,
            train_reward_min=np.min(train_rewards),
            train_reward_mean=np.mean(train_rewards),
            train_reward_med=np.median(train_rewards),
            train_reward_max=np.max(train_rewards),
            train_top_5_reward_avg=np.mean(np.sort(train_rewards)[-5:]),
            evaluate_reward_min=np.min(evaluate_rewards),
            evaluate_reward_mean=np.mean(evaluate_rewards),
            evaluate_reward_med=np.median(evaluate_rewards),
            evaluate_reward_max=np.max(evaluate_rewards),
            avg_timesteps_train=np.mean(
                [result['timesteps_total'] for result in results]),
            avg_timesteps_evaluate=np.mean(
                [result['timesteps_total'] for result in evaluate_results]),
            eval_max_video=evaluate_videos[np.argmax(evaluate_rewards)],
            eval_min_video=evaluate_videos[np.argmax(evaluate_rewards)],
        )

        #self.add_videos_to_summary(results, summary)
        return summary



# Using ES for optimization
def evolution_strategy_train(env, agent, hyperparams):
    # TODO
    return