
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

# Using GA for optimization
def genetic_algorithm_train(env, agent, hyperparams):
    population = [agent.clone() for _ in range(hyperparams.population_size)]
    hof = []  # Hall of Fame

    for generation in range(MAX_GENERATIONS):
        fitness_scores = []
        for individual in population:
            fitness = evaluate_individual(env, individual, hof, hyperparams.hof_evaluations)
            fitness_scores.append(fitness)

        # Sort by fitness
        ranked_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
        hof = update_hall_of_fame(hof, ranked_population, max_size=10)

        # Generate new population
        population = generate_new_population(ranked_population, hyperparams.noise_std)

# Fitness evaluation for GA
def evaluate_individual(env, individual, hof, hof_evaluations):
    fitness = 0
    for _ in range(hof_evaluations):
        opponent = np.random.choice(hof) if hof else None
        fitness += run_episode(env, individual)
    return fitness / hof_evaluations

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
