import argparse
import torch
import os
from agent import Agent
from genetic_algorithm import genetic_algorithm_train
from evolutionary_strategy import evolution_strategy_train
from utils.game_logic_functions import initialize_env
from supersuit import frame_stack_v1, resize_v1, frame_skip_v0, agent_indicator_v0
from utils.game_logic_functions import create_agent, play_game
from utils.utils_policies import RandomPolicy, PeriodicPolicy, AlwaysFirePolicy
from utils.utils_pth_and_plots import create_output_dir, load_agent_for_testing



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a DNN agent using GA or ES")
    
    # General arguments
    parser.add_argument("--algorithm", type=str, choices=["GA", "ES"], default="GA", 
                        help="Choose the algorithm to use: GA (Genetic Algorithm) or ES (Evolution Strategies)")
    parser.add_argument("--generations", type=int, default=100, 
                        help="Number of generations for the evolutionary process")
    parser.add_argument("--population", type=int, default=10, 
                        help="Population size for the evolutionary algorithm")
    parser.add_argument("--hof_size", type=int, default=20, 
                        help="Size of the Hall of Fame")
    parser.add_argument("--game", type=str, choices=["boxing_v2", "pong_v3", "simple_adversary_v3"], default="pong_v3",
                        help="Choose the Atari games to test with")
    parser.add_argument("--initial_mutation_power", type=float, default=0.05, 
                        help="Initial value for the mutation power, i.e. noise standard deviation for weight mutation")
    parser.add_argument("--learning_rate", type=float, default=0.1, 
                        help="Learning rate")
    parser.add_argument("--max_timesteps_per_episode", type=int, default=None, 
                        help="Total timesteps per episode")
    parser.add_argument("--max_evaluation_steps", type=int, default=None, 
                        help="Maximum steps for evaluation")
    parser.add_argument("--elites_number", type=int, default=2, 
                        help="Number of elites for each generation")          
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode for detailed logging")
    parser.add_argument("--train", action="store_true",
                        help="Train the models")
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering during training")
    parser.add_argument("--test", action="store_true",
                        help="Test the models")
    parser.add_argument("--env_mode", type=str, choices=["AEC"], default="AEC", 
                        help="Choose the environment mode. Currently, only AEC is supported")
    parser.add_argument("--precision", type=str, choices=["float32", "float16"], default="float32",
                        help="Specify the precision for computations: float32 or float16")
    parser.add_argument("--save", action="store_true",
                        help="Save the models")
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable Dynamic Mutation Power via Reward Feedback")
    parser.add_argument("--max_mutation_power", type=float, default=0.2,
                        help="If adaptive is true, it specifies the max mutation power reachable")
    parser.add_argument("--min_mutation_power", type=float, default=0.001,
                        help="If adaptive is true, it specifies the max mutation power reachable")
    parser.add_argument("--fitness_sharing", action="store_true",
                        help="If true, reduce Genetic Drift with the Niching technique of fitness sharing")
    parser.add_argument("--ES_model_to_test", type=str, default=None,
                        help="If testing ES, choose the model to test from CLI")
    parser.add_argument("--GA_hof_to_test", type=str, default=None,
                        help="If testing GA, choose the model to test from CLI")
    parser.add_argument("--play_against_yourself", action="store_true",
                        help="If true, the model plays against itself. If false, it plays against a dummy (RandomPolicy)")
    parser.add_argument("--average_window", type=int, default=None,
                        help="Choose the window for the running average")

    return parser.parse_args()


# Define hyperparameters
class Args:
    def __init__(self, args):

        # Hyperparameters
        self.algorithm = args.algorithm
        self.generations = args.generations
        self.population = args.population
        self.hof_size = args.hof_size
        self.game = args.game
        self.mutation_power = args.initial_mutation_power
        self.learning_rate = args.learning_rate
        self.max_timesteps_per_episode = args.max_timesteps_per_episode
        self.max_evaluation_steps = args.max_evaluation_steps
        self.elites_number = args.elites_number
        self.adaptive = args.adaptive
        self.max_mutation_power = args.max_mutation_power
        self.min_mutation_power = args.min_mutation_power
        self.fitness_sharing = args.fitness_sharing

        # clearly, these are not hyperparam, but it's easy to have everything inside an object
        self.debug = args.debug
        self.train = args.train
        self.test = args.test
        self.render = args.render
        self.env_mode = args.env_mode
        self.precision = args.precision
        self.save = args.save
        self.ES_model_to_test = args.ES_model_to_test
        self.GA_hof_to_test = args.GA_hof_to_test
        self.play_against_yourself = args.play_against_yourself
        if args.average_window != None:
            if args.average_window > args.generations:
                raise ValueError(f"The average window must be lower than the total number of generations!")
            else:
                self.average_window = args.average_window
        if args.average_window == None:
            #self.average_window = args.generations / 10
            self.average_window = 50


    def print_attributes(self, args): 
        # Print all attributes except `input_channel`
        attributes = vars(self)  # Get all attributes as a dictionary
        for attr, value in attributes.items():
                if attr != "elites_number" or args.algorithm=="GA":
                    if attr != "learning_rate" or args.algorithm=="ES":
                        if attr != "ES_model_to_test" or (args.algorithm == "ES" and args.test and not args.train):
                            if attr != "GA_hof_to_test" or (args.algorithm == "GA" and args.test and not args.train):
                                if attr != "play_against_yourself" or args.test:
                                    if (attr != "max_evaluation_steps" and attr != "max_timesteps_per_episode") or args.game != "simple_adversary_v3":
                                        if attr not in ["max_mutation_power", "min_mutation_power"] or args.adaptive:
                                            print(f"{attr.replace('_', ' ').capitalize()}: {value}")


def main():
    
    args = parse_arguments()

    print("\nHyperparameters and Parameters:")
    args = Args(args)
    args.print_attributes(args)
    print("\n")

    output_dir = create_output_dir(args)

    if args.train:

        print("Starting Training...")

        env = initialize_env(args)

        if args.algorithm == "GA":
            hof = genetic_algorithm_train(env, env.agents[0], args, output_dir)
        elif args.algorithm == "ES":
            agent = evolution_strategy_train(env, env.agents[0], args, output_dir)
        else:
            print("Unknown algorithm. Exiting.")
            return

        print("Training completed.")
        
        env.close()


    if args.test:

        print("Starting Testing...")
        
        env = initialize_env(args)

        agent = load_agent_for_testing(args, env)
        print(f"agent = {agent}")

        total_rewards = 0
        test_episodes = 10

        if args.game == "simple_adversary_v3": # cooperative task
    
            total_rewards1 = 0
            total_reward2 = 0

            for episode in range(test_episodes):

                reward1, reward2 = play_game(env=env, player1=agent.model,
                                                            player2=agent.model, adversary=RandomPolicy(env.action_space(env.agents[0]).n),
                                                            args=args, eval=True)
                total_rewards1 += reward1
                total_reward2 += reward2

            avg_reward1 = total_rewards1 / test_episodes
            avg_reward2 = total_reward2 / test_episodes

            print(f"\n Average Reward over {test_episodes}: {avg_reward1} and {avg_reward2}")

        else: 
            
            # atari game
            
            total_rewards = 0

            for episode in range(test_episodes):

                if args.play_against_yourself:
                    reward, _ = play_game(env=env, player1=agent.model,
                                                player2=agent.model, 
                                                args=args, eval=True)
                else:
                    reward, _ = play_game(env=env, player1=agent.model,
                                                player2=RandomPolicy(env.action_space(env.agents[0]).n), 
                                                args=args, eval=True)

                total_rewards += reward

            avg_reward = total_rewards / test_episodes
        
            print(f"\n Average Reward over {test_episodes}: {avg_reward}")


        print("Testing completed.")  
        
        env.close()
            
           
if __name__ == "__main__":
    main()
