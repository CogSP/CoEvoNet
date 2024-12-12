# A Coevolutionary Approach to Deep Multi-Agent Reinforcement Learning

This repository implements a coevolutionary approach to training agents for multi-agent reinforcement learning environments. Inspired by **Deep Neuroevolution** and **Coevolutionary Algorithms**, we combine genetic algorithms (GA) and evolution strategies (ES) to train deep neural networks for challenging multi-agent decision-making problems.

## Table of Contents
- [Features](#features)
- [Requirements and installation](#requirements)
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Environments](#environments)
- [Results](#results)
- [References](#references)

---

## Features
- **Coevolutionary Training**: Employs genetic algorithms and evolution strategies to evolve neural networks in a multi-agent setting.
- **Multi-Agent Support**: Designed for PettingZoo environments, supporting both cooperative and competitive scenarios.
- **Pre-trained Models**: Includes support for integrating pre-trained models trained with frameworks like `Stable-Baselines3`.
- **Customizable**: Easily configure environment wrappers, neural network architectures, and hyperparameters.
- **Benchmarks**: Tested on PettingZoo's Atari games and MPE.

---

## Requirements and Installation
- Python: we used version 3.9
- Install all the required dependencies using:
```bash
./install.sh
```

## Usage
- ```train_ES``` and ```train_GA``` to train
- ```test_ES``` and ```test_GA``` to test the trained models.

Please inspect the file if you want to check and modify config and hyperparameters.


## Algorithms
 
### Coevolutionary Evolution Strategies (Co-ES)

Adapts Evolution Strategies (ES) for multi-agent environments. Agents evolve by optimizing fitness functions tailored to competitive/cooperative tasks.

### Coevolutionary Genetic Algorithms (Co-GA)

Uses mutation, crossover, and selection to evolve agents. Incorporates a Hall of Fame (HoF) to maintain strong adversaries for evaluation.


## Environments
Supported Environments
The framework supports the following PettingZoo Games:
- simple_adversary_v3: A cooperative-competitive MPE task involving good agents and adversaries.
- pong_v3 and boxing_v2: Competitive Atari-based tasks.


# Results
TODO

# References

- "Coevolutionary Reinforcement Learning in Multi-Agent Environments",
by Daan Klijn and A.E. Eiben, Vrije Universiteit Amsterdam.
