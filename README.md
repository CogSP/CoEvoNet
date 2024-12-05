NOTE: WE ARE USING PYTHON 3.9 VERSION

- The DeepQN model (neural network) serves as the agent that interacts with the environment.


- In GA: We mutate or crossover the weights of the neural network for evolutionary selection.


- In ES: We apply noise to the weights of the neural network, simulating mutations, and update the network weights based on the fitness scores of the agent's actions.

This approach treats the agent (DNN) as an evolving entity whose weights are optimized over generations using evolutionary algorithms instead of traditional reinforcement learning methods. The fitness of each agent is determined by how well it performs in the environment, and the weights are evolved by the evolutionary strategy.


# Running the program

First, install all the dependencies running
```bash
./install.sh
```
Run the script from the command line and specify the desired algorithm and hyperparameters. For instance:
```bash
python main.py --algorithm GA --generations 50 --population 200 --initial_mutation_power 0.05
```