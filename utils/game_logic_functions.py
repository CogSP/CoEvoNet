import os
import importlib
import torch
import numpy as np
from MPE.mpe_agent import MPEAgent
from Atari.atari_agent import AtariAgent
from utils.utils_policies import RandomPolicy


def diversity_penalty(individual_weights, population_weights, args, sigma=None):

    distances = []
    for individual in population_weights:
        distance = np.linalg.norm(individual - individual_weights)
        distances.append(distance)
    
    distances = np.array(distances)

    # Dynamically set sigma if not provided
    # we can think of plotting sigma too
    if sigma is None:
        sigma = np.mean(distances)  

    if args.debug:
        print("Distance Range:", distances.min(), distances.max(), "Sigma:", sigma)

    # if distances > sigma, then sharing function = 0
    sharing_function = np.maximum(0, 1 - distances / sigma)

    diversity_scores = np.sum(sharing_function)
    
    if args.debug:
        print(f"diversity_scores = {diversity_scores}")
        
    return diversity_scores



def initialize_env(args):
    """Initialize the environment """
    env = None
    if args.game == "simple_adversary_v3":
        mpe_game_module = importlib.import_module(f"pettingzoo.mpe.{args.game}")
        env = mpe_game_module.env(render_mode="human" if args.render else None)
    else:
        atari_game_module = importlib.import_module(f"pettingzoo.atari.{args.game}")
        env = atari_game_module.env(render_mode="human" if args.render else None, obs_type="grayscale_image")
        env = frame_skip_v0(env, 4)
        env = resize_v1(env, 84, 84)
        env = frame_stack_v1(env, 4)
        env = agent_indicator_v0(env)
    env.reset(seed=1938214)
    return env


def create_agent(env, args):
    if args.game == "simple_adversary_v3":
        return MPEAgent(env, args)
    elif args.game == "pong_v3" or args.game == "boxing_v2":
        return AtariAgent(env, args)
    else:
        raise ValueError(f"Unsupported game type: {args.game}")


def preprocess_observation(obs, args):

    obs = torch.from_numpy(obs) # Ensure it's of the correct dtype
    if args.precision == "float16":
        obs = obs.to(torch.float16)
    else:
        obs = obs.to(torch.float32)

    if args.game == "simple_adversary_v3":
        # TODO
        pass
    else:  # atari games, that deal with images
        obs = obs.permute(2, 0, 1)
        obs = obs.unsqueeze(0) 
    return obs



def play_atari(player1, player2, args):

    rewards = {"first_0": 0, "second_0": 0}
    timesteps = 0
    timesteps_limit = args.max_evaluation_steps if eval else args.max_timesteps_per_episode
    done = False

    for agent in env.agent_iter():

        obs = env.observe(agent)

        obs = preprocess_observation(obs, args)
        
        if agent == "first_0":
            action = player1.determine_action(obs)
        elif agent == "second_0":
            action = player2.determine_action(obs)  
        else:
            raise ValueError(f"Unknown Agent during play_game: {agent}")
        
        env.step(action)    
        
        _, reward, termination, truncation, _ = env.last()
        
        rewards[agent] += reward
        
        timesteps += 1
    
        if timesteps_limit is not None and timesteps >= timesteps_limit:
            break

        if termination or truncation:
            break

         
    return rewards["first_0"], rewards["second_0"]



def play_MPE(env, player1, player2, adversary, args):

    rewards = {"agent_0": 0, "agent_1": 0, "adversary_0": 0}
    timesteps = 0
    timesteps_limit = args.max_evaluation_steps if eval else args.max_timesteps_per_episode
    done = False

    for agent in env.agent_iter():

        obs = env.observe(agent)

        obs = preprocess_observation(obs, args)
        if agent == "agent_0":
            action = player1.determine_action(obs)
        elif agent == "agent_1":
            action = player2.determine_action(obs)  
        elif agent == "adversary_0":
            action = adversary.determine_action(obs)
        else:
            raise ValueError(f"Unknown Agent during play_game: {agent}")
        
        env.step(action)    
        
        _, reward, termination, truncation, _ = env.last()
        
        rewards[agent] += reward
        
        timesteps += 1
    
        if timesteps_limit is not None and timesteps >= timesteps_limit:
            break

        if termination or truncation:
            break
         
    return rewards["agent_0"], rewards["agent_1"], rewards["adversary_0"]


def play_game(env, player1, player2, adversary=None, args=None, eval=False):
    """Play a game using the weights of two players in the PettingZoo environment."""
    env.reset()

    if args.game == "simple_adversary_v3":
        if adversary is None:
            adversary = RandomPolicy(env.action_space(env.agents[0]).n)
        rw_p1, rw_p2, rw_adv = play_MPE(env, player1, player2, adversary, args)
        return rw_p1, rw_p2
    else:
        rw_p1, rw_p2 = play_atari(env, player1, player2, args) 
        return rw_p1, rw_p2
    