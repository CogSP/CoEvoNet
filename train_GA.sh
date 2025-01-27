#!/bin/bash

python3.9 main.py \
    --algorithm=GA \
    --train \
    --save \
    --generations=500 \
    --population=20 \
    --hof_size=3 \
    --elites_number=5 \
    --precision=float32 \
    --game=simple_adversary_v3 \
    --max_timesteps_per_episode=400 \
    --max_evaluation_steps=400 \
    --initial_mutation_power_agent_0=0.005 \
    --initial_mutation_power_agent_0=0.005 \
    --initial_mutation_power_agent_0=0.005 \
    --adaptive \
    --max_mutation_power=0.7 \
    --min_mutation_power=0.0001 \
    --fitness_sharing