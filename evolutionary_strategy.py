import torch
import numpy as np
from deepqn import DeepQN
from genetic_algorithm import play_game
from agent import Agent

POPULATION_SIZE = 50
MUTATION_POWER = 0.05
LEARNING_RATE = 0.1
TIMESTEPS_TOTAL = 0
EPISODES_TOTAL = 0
GENERATION = 0
MAX_EVALUATION_STEPS = 500
INPUT_CHANNEL = 3
N_ACTIONS = None  # Not used
ELITES_NUMBER = 1  # Numero di soluzioni da mantenere come elite

# Hall of Fame (memorizza le migliori soluzioni)
hall_of_fame = []

def evaluate_current_weights(weights, env):
    """Evaluate the current weights against a random policy."""
    total_reward = 0
    for _ in range(10):  # Evaluate over multiple episodes
        agent = DeepQN(input_channels=INPUT_CHANNEL, n_actions=env.action_space.n)
        agent.set_weights(weights)
        total_reward += run_episode(env, agent)
    return total_reward / 10

def run_episode(env, agent):
    """Run an episode and return the total reward."""
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = agent.determine_action(state_tensor)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    return total_reward

def mutate_weights(base_weights, mutation_power):
    """Apply Gaussian noise to the weights."""
    elite = Agent()
    elite.set_weights(base_weights)
    opponent = Agent()
    opponent.set_weights(base_weights)
    perturbations = opponent.mutate(args.mutation_power)

    _, oponent_reward1, ts1 = play_game(elite, opponent)
    oponent_reward2, _, ts2 = play_game(opponent, elite)
    
    total_reward = np.mean([oponent_reward1, oponent_reward2])
    noise = perturbations

    return total_reward, noise


def compute_weight_update(noises, rewards, learning_rate, mutation_power):
    """Compute the weight update based on the rewards and noises."""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) if np.std(rewards) > 0 else 1.0
    normalized_rewards = (rewards - mean_reward) / std_reward
    
    weights_update = learning_rate / (len(noises) * mutation_power) * np.dot(np.array(noises).T, normalized_rewards)

    return weights_update

def evolution_strategy_train(env, agent, max_generations):
    """Train the agent using Evolution Strategies with Elitism and Hall of Fame."""
    global TIMESTEPS_TOTAL, EPISODES_TOTAL, GENERATION, hall_of_fame

    base_weights = agent.get_weights()
    elite_weights = None
    elite_reward = float('-inf')

    all_rewards = []
    evaluate_rewards = []

    for gen in range(max_generations):
        noises = []
        rewards = []

        # Step 1: Generate population by mutating weights
        for _ in range(POPULATION_SIZE):
            total_reward, noise = mutate_weights(base_weights, MUTATION_POWER)
            noises.append(noise)
            rewards.append(total_reward)

        """
        DA TESTARE SE MIGLIORA !!
        
        # Step 2: Elitism - Retain the best individual
        max_reward_idx = np.argmax(rewards)
        if rewards[max_reward_idx] > elite_reward:
            elite_reward = rewards[max_reward_idx]
            elite_weights = {key: base_weights[key] + noises[max_reward_idx][key] for key in base_weights}
        """

        # Step 3: Compute weight update
        weight_update = compute_weight_update(noises, rewards, LEARNING_RATE, MUTATION_POWER)
        base_weights = base_weights + weight_update
        agent.set_weights(base_weights)

        # Step 4: Evaluate current weights
        evaluation_reward = evaluate_current_weights(base_weights, env)
        evaluate_rewards.append(evaluation_reward)

        # Update metrics
        TIMESTEPS_TOTAL += sum(rewards)
        EPISODES_TOTAL += POPULATION_SIZE
        GENERATION += 1

        all_rewards.extend(rewards)

        # Log results
        print(f"Generation {gen + 1}:")
        print(f"  Train Reward - Mean: {np.mean(rewards):.2f}, Max: {np.max(rewards):.2f}")
        print(f"  Evaluation Reward: {evaluation_reward:.2f}")
        print(f"  Best Reward in Hall of Fame: {hall_of_fame[0][0]:.2f}")

    # Generate summary
    summary = dict(
        timesteps_total=TIMESTEPS_TOTAL,
        episodes_total=EPISODES_TOTAL,
        train_reward_min=np.min(all_rewards),
        train_reward_mean=np.mean(all_rewards),
        train_reward_max=np.max(all_rewards),
        train_top_5_reward_avg=np.mean(np.sort(all_rewards)[-5:]),
        evaluate_reward_min=np.min(evaluate_rewards),
        evaluate_reward_mean=np.mean(evaluate_rewards),
        evaluate_reward_med=np.median(evaluate_rewards),
        evaluate_reward_max=np.max(evaluate_rewards),
        avg_timesteps_train=TIMESTEPS_TOTAL / EPISODES_TOTAL,
        avg_timesteps_evaluate=TIMESTEPS_TOTAL / max(len(evaluate_rewards), 1),
        best_reward_hof=hall_of_fame[0][0],
        best_weights_hof=hall_of_fame[0][1]
    )

    return summary
