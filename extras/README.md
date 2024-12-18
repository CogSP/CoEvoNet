# Motivation

We have tried to conduct experiments using SB pre-trained model, in order to measure the performances of our model interacting with SB models.
Unfortunately we have found that SB3 is made for single agent, so it was not possible to train, for instance, A2C with the role of adversary in MPE Simple Adversary v3. Indeed, agent and adversary have different state and action space in that environment.