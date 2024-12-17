import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np


def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight.data, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)


class PPOPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        
        self.init_weights()

    def init_weights(self):
        # Orthogonal initialization of the hidden layers with gain = sqrt(2)
        # This is a common initialization approach for PPO.
        orthogonal_init(self.fc1, gain=np.sqrt(2))
        orthogonal_init(self.fc2, gain=np.sqrt(2))
        
        # For the policy head, use a smaller gain for stable initial policy.
        orthogonal_init(self.policy_head, gain=0.01)
        
        # For the value head, also use a smaller gain.
        orthogonal_init(self.value_head, gain=1.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

    def get_action_and_value(self, obs, deterministic=False):
        obs_t = obs.clone().detach().unsqueeze(0).to(torch.float32)
        logits, value = self.forward(obs_t)
        action_probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()


class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4):
        self.policy = PPOPolicy(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.ppo_update_epoch = 10
        self.minibatch_size = 64
        self.vf_coeff = 0.5 
        self.ent_coeff = 0.01

    def get_action_and_value(self, obs, deterministic=False):
        return self.policy.get_action_and_value(obs, deterministic)

    def update(self, obs, actions, old_log_probs, returns, advantages, clip_range):   
        # Detach all these tensors to ensure they're treated as constants.
        obs = obs.detach()
        actions = actions.detach()
        old_log_probs = old_log_probs.detach()
        returns = returns.detach()
        advantages = advantages.detach()

        dataset_size = len(obs)
        indices = np.arange(dataset_size)

        policy_losses = []
        value_losses = []
        entropies = []
        
        for _ in range(self.ppo_update_epoch):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                logits, values = self.policy.forward(batch_obs)
                action_probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                value_loss = ((values.squeeze() - batch_returns)**2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + self.vf_coeff * value_loss - self.ent_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)
