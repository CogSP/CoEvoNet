import torch
import numpy as np

ELITES_NUMBER = 5
POPULATION_SIZE = 10
MUTATION_POWER = 0.05



def evaluate_current_weights(self, best_mutation_weights):
    """ Send the weights to a number of workers and ask them to evaluate the weights. """
    evaluate_jobs = []
    for i in range(self.config['evaluation_games']):
        worker_id = i % self.config['num_workers']
        evaluate_jobs += [self._workers[worker_id].evaluate.remote(
            best_mutation_weights)]
    evaluate_results = ray.get(evaluate_jobs)
    return evaluate_results

def increment_metrics(self, results):
    """ Increment the total timesteps, episodes and generations. """
    self.timesteps_total += sum([result['timesteps_total'] for result in results])
    self.episodes_total += len(results)
    self.generation += 1


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


def play_game(self, player1, player2, recorder=None, eval=False):
    """ Play a game using the weights of two players. """
    obs = self.env.reset()
    reward1 = 0
    reward2 = 0
    # play and return the rewards and ts
    return 


def evaluate_mutations(self, elite, oponent, record=False, mutate_oponent=True):
    """ Mutate the inputted weights and evaluate its performance against the inputted oponent. """
    self.elite.set_weights(elite)
    self.oponent.set_weights(oponent)
    if mutate_oponent:
        self.oponent.mutate(MUTATION_POWER)
    elite_reward1, oponent_reward1, ts1 = self.play_game(
        self.elite, self.oponent)
    oponent_reward2, elite_reward2, ts2 = self.play_game(
        self.oponent, self.elite)
    total_elite = elite_reward1 + elite_reward2
    total_oponent = oponent_reward1 + oponent_reward2
    return {
        'oponent_weights': self.oponent.get_weights(),
        'score_vs_elite': total_oponent,
        'timesteps_total': ts1 + ts2,
}



def genetic_algorithm_train(env, agent, hyperparams):
    
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


    elites = [create num_elites networks]

    # do things with VBN

    hof = [elites[i].get_weights() for i in
                range(ELITES_NUMBER)]
    
    for gen in range(MAX_GENERATION):
        
        # Evaluate mutations vs first hof
        for i in range(POPULATION_SIZE):
            elite_id = i % ELITES_NUMBER
            should_mutate = (i > ELITES_NUMBER) # elitarism (?)
            results += [evaluate_mutations.remote(hof[-1], elites[elite_id].get_weights(), mutate_oponent=should_mutate)]


        # Evaluate vs other hof
        for j in range(len(hof) - 1):
            for i in range(POPULATION_SIZE):
                results += [evaluate_mutations.remote(hof[-2 - j], results[i]['oponent_weights'], mutate_oponent=False)]

        rewards = []
        print(len(results))
        for i in range(POPULATION_SIZE):
            total_reward = 0
            for j in range(ELITES_NUMBER):
                reward_index = POPULATION_SIZE * j + i
                total_reward += results[reward_index]['score_vs_elite']
            rewards.append(total_reward)

        best_mutation_id = np.argmax(rewards)
        best_mutation_weights = results[best_mutation_id]['oponent_weights']
        print(f"Best mutation: {best_mutation_id} with reward {np.max(rewards)}")

        #self.try_save_winner(best_mutation_weights)
        hof.append(best_mutation_weights)

        new_elite_ids = np.argsort(rewards)[-ELITES_NUMBER:]
        print(f"TOP mutations: {new_elite_ids}")
        for i, elite in enumerate(elites):
            mutation_id = new_elite_ids[i]
            elite.set_weights(results[mutation_id]['oponent_weights'])

        # Evaluate best mutation vs random agent
        evaluate_results = evaluate_current_weights(best_mutation_weights)
        evaluate_rewards = [result['total_reward'] for result in evaluate_results]

        train_rewards = [result['score_vs_elite'] for result in results]
        evaluate_videos = [result['video'] for result in evaluate_results]

        self.increment_metrics(results)

        summary = dict(
            timesteps_total=self.timesteps_total,
            episodes_total=self.episodes_total,
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
        weights = agent.get_weights()
        for generation in range(MAX_GENERATIONS):
            noise_samples = [np.random.normal(0, hyperparams.noise_std, size=weights['model'].shape) for _ in range(hyperparams.population_size)]
            fitness_scores = []
            
            for noise in noise_samples:
                perturbed_weights = weights + noise
                agent.set_weights(perturbed_weights)
                fitness = run_episode(env, agent)
                fitness_scores.append(fitness)
            
            # Update weights with the best fitness scores
            weights += LEARNING_RATE * np.mean(fitness_scores, axis=0)
