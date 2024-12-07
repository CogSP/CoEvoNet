import torch
import numpy as np
from deepqn import DeepQN
from genetic_algorithm import play_game, RandomPolicy, plot_rewards
from agent import Agent
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

POPULATION_SIZE = 50
MUTATION_POWER = 0.05
LEARNING_RATE = 0.1
TIMESTEPS_TOTAL = 0
EPISODES_TOTAL = 0
GENERATION = 0
MAX_EVALUATION_STEPS = 500
INPUT_CHANNEL = 3
ELITES_NUMBER = 1  # Numero di soluzioni da mantenere come elite


def create_es_model_dir(args):
    """Create a directory for saving models based on hyperparameters."""
    dir_name = f"ES_models/gens{args.generations}_pop{args.population}_hof{args.hof_size}_game{args.atari_game}_mut{args.mutation_power}_lr{args.learning_rate}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def save_model(obj, file_path):
    """Save any object to a file."""
    torch.save(obj, file_path)

def evaluate_current_weights(weights, env, args):
    """Evaluate the current weights against a random policy."""
    total_reward = 0
    for _ in tqdm(range(10), desc="Evaluating weights", leave=False):  # Evaluate over multiple episodes
        agent = DeepQN(input_channels=INPUT_CHANNEL, n_actions=env.action_space(env.agents[0]).n)
        agent.set_weights_ES(flat_weights=weights, args=args)
        reward, _, ts = play_game(env=env, player1=agent,
                                    player2=RandomPolicy(env.action_space(env.agents[0]).n), 
                                    args=args, eval=True)
    return total_reward / 10


"""
def run_episode(env, agent):
    #Run an episode and return the total reward.
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
"""

def mutate_weights(env, base_weights, args):
    """Apply Gaussian noise to the weights."""
    elite = Agent(input_channels=args.input_channels, n_actions=env.action_space(env.agents[0]).n, precision=args.precision)
    elite.model.set_weights_ES(flat_weights=base_weights, args=args)
    opponent = Agent(input_channels=args.input_channels, n_actions=env.action_space(env.agents[0]).n, precision=args.precision)
    opponent.model.set_weights_ES(flat_weights=base_weights, args=args)
    perturbations = opponent.mutate_ES(args)
   
    _, oponent_reward1, ts1 = play_game(env=env, player1=elite.model, player2=opponent.model, args=args)
    oponent_reward2, _, ts2 = play_game(env=env, player1=opponent.model, player2=elite.model, args=args)
    
    total_reward = np.mean([oponent_reward1, oponent_reward2])
    noise = perturbations

    return total_reward, noise


def compute_weight_update(noises, rewards, args):
    """Compute the weight update based on the rewards and noises."""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) if np.std(rewards) > 0 else 1.0
    normalized_rewards = (rewards - mean_reward) / std_reward

    weights_update = args.learning_rate / (len(noises) * args.mutation_power) * np.dot(np.array(noises).T, normalized_rewards)

    return weights_update

def evolution_strategy_train(env, agent, args):
    """Train the agent using Evolution Strategies with Elitism and Hall of Fame."""


    # Initialize directories and variables for saving
    model_dir = create_es_model_dir(args)
    agent_file = os.path.join(model_dir, "agent.pth")
    rewards_plot_file = os.path.join(model_dir, "rewards_plot.png")

    agent = Agent(input_channels=args.input_channels, n_actions=env.action_space(env.agents[0]).n, precision=args.precision)
    base_weights = agent.model.get_perturbable_weights()
    elite_weights = None
    elite_reward = float('-inf')

    all_rewards = []
    evaluate_rewards = []
    rewards_over_generations = []

    for gen in tqdm(range(args.generations), desc="Training Generations"):
        noises = []
        rewards = []

        # Step 1: Generate population by mutating weights
        for _ in tqdm(range(args.population), desc=f"Generation {gen} - Mutating", leave=False):
            total_reward, noise = mutate_weights(env, base_weights, args)
            noises.append(noise)
            rewards.append(total_reward)

      
        weight_update = compute_weight_update(noises, rewards, args)
        base_weights = base_weights + weight_update
        agent.model.set_weights_ES(flat_weights=base_weights, args=args)

        evaluation_reward = evaluate_current_weights(base_weights, env, args)
        evaluate_rewards.append(evaluation_reward)
        rewards_over_generations.append(evaluation_reward)


        all_rewards.extend(rewards)


        # Adaptive mutation power
        args.mutation_power = max(0.01, args.mutation_power * 0.95)  # Reduce over generations

        # Save Hall of Fame and plot rewards at each generation
        save_model(agent, agent_file)
        plot_rewards(rewards_over_generations, rewards_plot_file)



    return agent
