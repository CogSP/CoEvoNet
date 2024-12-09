#!/bin/bash

python3.9 main.py \
    --train \
    --save \
    --generations=500 \
    --population=200 \
    --hof_size=3 \
    --elites_number=5 \
    --precision=float16 \
    --atari_game=pong_v3 \
    --algorithm=GA \
    --max_timesteps_per_episode=1000 \
    --max_evaluation_steps=1000 \
    --initial_mutation_power=0.005
