from pettingzoo.mpe import simple_adversary_v3
from PPO.PPOagent import PPOAgent
from utils.game_logic_functions import preprocess_observation
import numpy as np
from PPO.PPOagent import PPOAgent
import torch
from tqdm import tqdm 
import os
import matplotlib.pyplot as plt

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for step in reversed(range(len(rewards))):
        next_value = values[step+1].item() if step+1 < len(values) else 0
        delta = rewards[step] + gamma * (0 if dones[step] else next_value) - values[step].item()
        gae = delta + gamma * lam * (0 if dones[step] else gae)
        advantages.insert(0, gae)
    returns = [a + v.item() for a, v in zip(advantages, values)]
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    return advantages, returns

def collect_experience(env, agents, args):
    transitions = {agent: {"obs": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": []}
                   for agent in env.agents}

    env.reset()
    timesteps = 0

    for agent in env.agent_iter():
        obs = env.observe(agent)
        obs = preprocess_observation(obs, args)

        action, log_prob, value = agents[agent].get_action_and_value(obs)
        transitions[agent]["obs"].append(obs)
        transitions[agent]["actions"].append(action)
        transitions[agent]["log_probs"].append(log_prob)
        transitions[agent]["values"].append(value)

        env.step(action)
        next_obs, reward, termination, truncation, _ = env.last()

        transitions[agent]["rewards"].append(reward)
        transitions[agent]["dones"].append(termination or truncation)

        timesteps += 1
        if timesteps >= args.max_timesteps_per_episode or termination or truncation:
            break

    return transitions

def moving_average(data, window_size=50):
    if len(data) < window_size:
        # Not enough data for a moving average, return as is
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def train_PPO():
    class Args:
        max_timesteps_per_episode = 200
        epochs = 1000
        gamma = 0.99
        lam = 0.95
        lr = 3e-4
        clip_range = 0.2
        precision = "float32"
        game = "simple_adversary_v3"
    args = Args()

    # Initialize environment: simple_adversary_v3
    env = simple_adversary_v3.env(max_cycles=args.max_timesteps_per_episode)
    env.reset()

    # Initialize one PPOAgent per agent
    agents = {}
    for agent in env.agents:
        obs_shape = env.observation_space(agent).shape[0]
        act_dim = env.action_space(agent).n
        agents[agent] = PPOAgent(obs_shape, act_dim, lr=args.lr)
        
    # Initialize dictionaries to store metrics for each agent
    all_metrics = {agent: {"policy_loss": [], "value_loss": [], "entropy": [], "avg_reward": []} 
                   for agent in env.agents}

    # Main training loop
    pbar = tqdm(range(args.epochs), desc="Training Progress", unit="epoch")
    for epoch in pbar:
        # Collect one episode of experience
        transitions = collect_experience(env, agents, args)
        
        # We'll aggregate metrics for display in the progress bar
        epoch_pl = []
        epoch_vl = []
        epoch_ent = []
        epoch_r = []

        # Perform PPO updates for each agent separately
        for agent in env.agents:
            obs = torch.tensor(np.array(transitions[agent]["obs"]), dtype=torch.float32)
            actions = torch.tensor(transitions[agent]["actions"], dtype=torch.int32)

            if len(transitions[agent]["log_probs"]) == 0:
                # No transitions collected for this agent, skip
                continue

            old_log_probs = torch.stack(transitions[agent]["log_probs"])
            values = torch.stack(transitions[agent]["values"])
            rewards = transitions[agent]["rewards"]
            dones = transitions[agent]["dones"]

            if len(rewards) == 0:  # In case something breaks or no steps collected
                continue

            # Compute advantages and returns
            advantages, returns = compute_gae(rewards, values, dones, gamma=args.gamma, lam=args.lam)

            
            # normalization of advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update the agent            
            pl, vl, ent = agents[agent].update(obs, actions, old_log_probs, returns, advantages, clip_range=args.clip_range)

            avg_reward = np.mean(rewards)

            # Store metrics
            all_metrics[agent]["policy_loss"].append(pl)
            all_metrics[agent]["value_loss"].append(vl)
            all_metrics[agent]["entropy"].append(ent)
            all_metrics[agent]["avg_reward"].append(avg_reward)


    # After training completes, save the models
    os.makedirs("PPO", exist_ok=True)
    for agent in env.agents:
        save_path = f"PPO/model_{agent}.pth"
        torch.save(agents[agent].policy.state_dict(), save_path)
        print(f"Saved model for {agent} at {save_path}")
        
    # Plot the metrics
    # We will create four subplots: policy loss, value loss, entropy, avg reward
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Metrics for Agents')
    
    # For convenience
    metric_keys = [("policy_loss", "Policy Loss"), ("value_loss", "Value Loss"), 
                   ("entropy", "Entropy"), ("avg_reward", "Average Reward")]
    axes_flat = axs.flat

    window_size = 50  # window size for moving average

    for (m_key, m_name), ax in zip(metric_keys, axes_flat):
        for agent in env.agents:
            data = all_metrics[agent][m_key]
            # Plot raw data
            ax.plot(data, label=f"{agent} (raw)", alpha=0.5)
            # Plot smoothed data
            smoothed_data = moving_average(data, window_size=window_size)
            ax.plot(range(window_size-1, window_size-1+len(smoothed_data)), smoothed_data, 
                    label=f"{agent} (MA)", linewidth=2)
        ax.set_title(m_name)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.savefig("PPO/PPOTrain.png")
    plt.close()

if __name__ == "__main__":
    train_PPO()
