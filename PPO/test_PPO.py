import torch
import numpy as np
import os
from pettingzoo.mpe import simple_adversary_v3
from PPO.PPOagent import PPOAgent
from utils.game_logic_functions import preprocess_observation

# Modify these paths if your models are stored elsewhere
MODEL_DIR = "PPO"  
AGENTS = ["agent_0", "agent_1", "adversary_0"]

def load_agents(env, model_dir, agents_list):
    """Load PPO agents with their trained policies."""
    agents = {}
    for agent_name in agents_list:
        obs_dim = env.observation_space(agent_name).shape[0]
        act_dim = env.action_space(agent_name).n
        agent = PPOAgent(obs_dim, act_dim)  # create a new PPOAgent instance

        model_path = os.path.join(model_dir, f"model_{agent_name}.pth")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        agent.policy.load_state_dict(state_dict)
        agent.policy.eval()  # Set the policy to evaluation mode
        agents[agent_name] = agent
    return agents


def run_episode(env, agents, args=None, max_steps=200):
    frames = []
    env.reset()

    for agent in env.agent_iter():
        obs = env.observe(agent)
        obs = preprocess_observation(obs, args)
        
        # Get action from the agent's policy
        action, _, _ = agents[agent].get_action_and_value(obs, deterministic=True)  


        env.step(action)



        termination, truncation = env.terminations[agent], env.truncations[agent]
        if termination or truncation:
            # Once the environment signals termination or truncation, break early
            break


def test_PPO():
    class Args:
        precision = "float16"
        game = "simple_adversary_v3"

    args = Args()


    # Initialize environment
    env = simple_adversary_v3.env(max_cycles=50, render_mode="human") 
    env.reset()

    # Load trained agents
    agents = load_agents(env, MODEL_DIR, AGENTS)

    # Run a test episode
    run_episode(env, agents, args=args, max_steps=200)



    env.close()
