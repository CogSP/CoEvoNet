#!/bin/bash

python3.9 main.py \
    --algorithm=ES \
    --train \
    --save \
    --generations=500 \
    --population=20 \
    --hof_size=1 \
    --precision=float32 \
    --game=simple_adversary_v3 \
    --max_timesteps_per_episode=400 \
    --max_evaluation_steps=400 \
    --learning_rate=0.1 \
    --fitness_sharing \
    --adaptive \
    --initial_mutation_power_agent_0=0.05 \
    --initial_mutation_power_agent_1=0.05 \
    --initial_mutation_power_adversary=0.05 \
    --max_mutation_power=0.5 \
    --min_mutation_power=0.001 \