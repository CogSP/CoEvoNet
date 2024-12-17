from pettingzoo.mpe import simple_adversary_v3
from PPO.PPOagent import PPOAgent
from utils.game_logic_functions import preprocess_observation
import numpy as np
from PPO.PPOagent import PPOAgent
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class RunningMeanStd:
    """Track running mean and std of observations."""
    def __init__(self, shape, epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        # x is a batch of observations: shape [N, obs_dim]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / new_count

        self.mean = self.mean + delta * batch_count / new_count
        self.var = M2 / new_count
        self.count = new_count

    def normalize(self, x):
        # x: single observation or batch of observations
        # returns normalized x
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

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

def collect_experience(env, agents, args, obs_rms_dict):
    transitions = {agent: {"obs": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": []}
                   for agent in env.agents}

    env.reset()
    timesteps = 0

    for agent in env.agent_iter():
        obs = env.observe(agent)
        
        # Preprocess (to torch)
        obs = preprocess_observation(obs, args)
        # Convert to numpy for normalization
        obs_np = obs.cpu().numpy()

        # Update running stats and normalize for this specific agent
        obs_rms = obs_rms_dict[agent]
        obs_rms.update(obs_np[None, :])  # Add batch dimension [1, obs_dim]
        obs_np = obs_rms.normalize(obs_np)

        # Convert back to torch
        obs = torch.from_numpy(obs_np).float()

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
        max_timesteps_per_episode = 2048
        epochs = 10000
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

    agents = {}
    obs_rms_dict = {}
    for agent in env.agents:
        obs_dim = env.observation_space(agent).shape[0]
        act_dim = env.action_space(agent).n
        agents[agent] = PPOAgent(obs_dim, act_dim, lr=args.lr)
        obs_rms_dict[agent] = RunningMeanStd(obs_dim)  # separate normalization per agent
        
    # Initialize dictionaries to store metrics for each agent
    all_metrics = {agent: {"policy_loss": [], "value_loss": [], "entropy": [], "avg_reward": []}
                   for agent in env.agents}

    # Early stopping parameters
    patience = 200
    best_avg_reward = -float("inf")
    no_improvement_count = 0

    pbar = tqdm(range(args.epochs), desc="Training Progress", unit="epoch")
    for epoch in pbar:
        transitions = collect_experience(env, agents, args, obs_rms_dict)

        epoch_pl = []
        epoch_vl = []
        epoch_ent = []
        epoch_r = []

        for agent in env.agents:
            
            obs = torch.stack(transitions[agent]["obs"], dim=0).to(torch.float32)
            actions = torch.tensor(transitions[agent]["actions"], dtype=torch.int32)

            
            old_log_probs = torch.stack(transitions[agent]["log_probs"])
            values = torch.stack(transitions[agent]["values"])
            rewards = transitions[agent]["rewards"]
            dones = transitions[agent]["dones"]

            
            rewards = np.array(rewards)
            reward_mean = rewards.mean()
            reward_std = rewards.std() + 1e-8
            rewards = (rewards - reward_mean) / reward_std

            # Compute advantages and returns
            advantages, returns = compute_gae(rewards, values, dones, gamma=args.gamma, lam=args.lam)

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            pl, vl, ent = agents[agent].update(obs, actions, old_log_probs, returns, advantages, clip_range=args.clip_range)
            avg_reward = np.mean(rewards)

            all_metrics[agent]["policy_loss"].append(pl)
            all_metrics[agent]["value_loss"].append(vl)
            all_metrics[agent]["entropy"].append(ent)
            all_metrics[agent]["avg_reward"].append(avg_reward)

            epoch_pl.append(pl)
            epoch_vl.append(vl)
            epoch_ent.append(ent)
            epoch_r.append(avg_reward)

        if len(epoch_pl) > 0:
            pbar.set_postfix({
                "Policy Loss": f"{np.mean(epoch_pl):.3f}",
                "Value Loss": f"{np.mean(epoch_vl):.3f}",
                "Entropy": f"{np.mean(epoch_ent):.3f}",
                "Avg Reward": f"{np.mean(epoch_r):.3f}"
            })

            current_avg_reward = np.mean(epoch_r)
            if current_avg_reward > best_avg_reward:
                best_avg_reward = current_avg_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print("Early stopping triggered: no improvement in average reward for", patience, "epochs.")
                break

    os.makedirs("PPO", exist_ok=True)
    for agent in env.agents:
        save_path = f"PPO/model_{agent}.pth"
        torch.save(agents[agent].policy.state_dict(), save_path)
        print(f"Saved model for {agent} at {save_path}")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Metrics for Agents')
    metric_keys = [("policy_loss", "Policy Loss"), ("value_loss", "Value Loss"),
                   ("entropy", "Entropy"), ("avg_reward", "Average Reward")]
    axes_flat = axs.flat

    window_size = 50

    for (m_key, m_name), ax in zip(metric_keys, axes_flat):
        for ag in env.agents:
            data = all_metrics[ag][m_key]
            ax.plot(data, label=f"{ag} (raw)", alpha=0.5)
            smoothed_data = moving_average(data, window_size=window_size)
            ax.plot(range(window_size-1, window_size-1+len(smoothed_data)), smoothed_data,
                    label=f"{ag} (MA)", linewidth=2)
        ax.set_title(m_name)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.savefig("PPO/PPOTrain.png")
    plt.close()

if __name__ == "__main__":
    train_PPO()
