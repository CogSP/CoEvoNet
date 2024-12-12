import importlib
from MPE.mpe_agent import MPEAgent
from Atari.atari_agent import AtariAgent
from utils.utils_policies import RandomPolicy
import torch
import os


def load_agent_for_testing(args):

    if args.algorithm == "GA":

        if args.GA_hof_to_test is None:
            print(f"Error: HoF file not specified. Please specify the HoF to test")
            return
            
        if not os.path.exists(args.GA_hof_to_test):
            print(f"Error: Model file {args.GA_hof_to_test} not found.")
            return

        print(f"Loading HoF from {args.GA_hof_to_test} for testing...")
        hof = torch.load(file_path)
        agent = create_agent(env, args)
        agent.set_weights(hof[-1])

        return agent

    elif args.algorithm == "ES":

        if args.ES_model_to_test is None:
            print(f"Error: Model file not specified. Please specify the agent to test")
            return
            
        if not os.path.exists(args.ES_model_to_test):
            print(f"Error: Model file {args.ES_model_to_test} not found.")
            return

        print(f"Loading Agent from {args.ES_model_to_test} for testing...")
        agent = torch.load(args.ES_model_to_test)

        return agent


def initialize_env(args):
    """Initialize the environment """
    env = None
    if args.game == "simple_adversary_v3":
        mpe_game_module = importlib.import_module(f"pettingzoo.mpe.{args.game}")
        env = mpe_game_module.env(render_mode="human" if args.render else None)
    else:
        atari_game_module = importlib.import_module(f"pettingzoo.atari.{args.game}")
        if args.env_mode == "AEC":
            env = atari_game_module.env(render_mode="human" if args.render else None, obs_type="grayscale_image")
            env = frame_skip_v0(env, 4)
            env = resize_v1(env, 84, 84)
            env = frame_stack_v1(env, 4)
            env = agent_indicator_v0(env)
        elif args.env_mode == "parallel":
            env = atari_game_module.parallel_env(render_mode="human" if args.render else None)
        else:
            raise ValueError("Invalid environment mode. Choose either 'AEC' or 'parallel'.")
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
        rw_p1, rw_p2, rw_adv = play_MPE(env, player1, player2, adversary, args)
        return rw_p1, rw_p2
    else:
        rw_p1, rw_p2 = play_atari(env, player1, player2, args) 
        return rw_p1, rw_p2
    