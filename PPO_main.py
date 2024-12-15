from PPO.train_PPO import train_PPO
from PPO.test_PPO import test_PPO
import argparse

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a DNN agent using GA or ES")
    
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_arguments()
    
    if args.train:
        train_PPO()
        
    if args.test:
        test_PPO()