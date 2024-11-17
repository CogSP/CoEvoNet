- The DeepQN model (neural network) serves as the agent that interacts with the environment.


- In GA: We mutate or crossover the weights of the neural network for evolutionary selection.


- In ES: We apply noise to the weights of the neural network, simulating mutations, and update the network weights based on the fitness scores of the agent's actions.

This approach treats the agent (DNN) as an evolving entity whose weights are optimized over generations using evolutionary algorithms instead of traditional reinforcement learning methods. The fitness of each agent is determined by how well it performs in the environment, and the weights are evolved by the evolutionary strategy.


# Running the program

Run the script from the command line and specify the desired algorithm and hyperparameters:

```bash
python main.py --algorithm GA --generations 50 --population 200 --noise_std 0.005
```

```bash
python main.py --algorithm ES --generations 50 --population 200 --noise_std 0.05
```