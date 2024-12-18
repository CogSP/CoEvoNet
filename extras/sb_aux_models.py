import argparse
from stable_baselines3 import A2C
from utils.game_logic_functions import initialize_env
import torch

# PettingZoo environments are multi-agent and not directly compatible with SB3
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from pettingzoo.utils.conversions import aec_to_parallel
from supersuit import pad_action_space_v0


def initialize_wrapped_env(args, agent_type="adversary"):
    """
    Initialize and wrap the PettingZoo environment for Stable-Baselines3 compatibility.

    Args:
        args: Parsed command-line arguments.
        agent_type: Type of agent ('adversary' or 'good_agent') to handle different observation spaces.

    Returns:
        A Gym-compatible single-agent environment.
    """
    # Initialize the PettingZoo environment
    env = initialize_env(args)

    # Convert AECEnv to ParallelEnv
    parallel_env = aec_to_parallel(env)

    # Trim observations based on the agent type
    if agent_type == "adversary":
        parallel_env = TrimObservationsWrapper(parallel_env, expected_size=8)
    elif agent_type == "good_agent":
        parallel_env = TrimObservationsWrapper(parallel_env, expected_size=10)

    # Pad actions to ensure consistency
    padded_env = pad_action_space_v0(parallel_env)

    # Wrap the PettingZoo environment to make it Gym-compatible
    vec_env = pettingzoo_env_to_vec_env_v1(padded_env)
    wrapped_env = concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    return wrapped_env


class TrimObservationsWrapper:
    """
    Custom wrapper to trim observations to a specific size.

    Args:
        env: The ParallelEnv to wrap.
        expected_size: The expected size of the observations.
    """
    def __init__(self, env, expected_size):
        self.env = env
        self.expected_size = expected_size

    def observe(self, agent):
        obs = self.env.observe(agent)
        return obs[:self.expected_size]

    def __getattr__(self, name):
        return getattr(self.env, name)


def train_model(args, agent_type="adversary"):
    """
    Train and save a model for a specific agent type.

    Args:
        args: Parsed command-line arguments.
        agent_type: Type of agent ('adversary' or 'good_agent') to train.
    """

    env = initialize_wrapped_env(args, agent_type=agent_type)

    # Debug: Check observation space size
    print(f"Training {agent_type} on observation space of shape: {env.observation_space(env.agents[0]).shape}")

    if args.model == "A2C":
        model = A2C("MlpPolicy", env, verbose=1)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    print(f"Training {args.model} for {agent_type} on {args.game} for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    save_path = f"utils/pretrained_aux_models/{args.model.lower()}_{args.game}_{agent_type}.pth"
    torch.save(model.policy.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train and save an RL model.")
    parser.add_argument("--model", type=str, choices=["A2C"], required=True,
                        help="The RL algorithm to use.")
    parser.add_argument("--game", type=str, choices=["simple_adversary_v3", "pong_v3", "boxing_v2"], required=True,
                        help="The environment to train on (e.g., simple_adversary_v3, pong_v3).")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="The number of timesteps to train for.")
    parser.add_argument("--render", action="store_false", default=False,
                        help="Rendering is always disabled in this case.")
    parser.add_argument("--agent_type", type=str, choices=["adversary", "good_agent"], default="adversary",
                        help="The type of agent to train (adversary or good_agent).")

    args = parser.parse_args()

    # Ensure render is always False
    args.render = False  

    # Train model for the specified agent type
    train_model(args, agent_type=args.agent_type)
