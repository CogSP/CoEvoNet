#!/bin/bash

python3.9 main.py \
    --train \
    --save \
    --generations=2 \
    --population=10 \
    --hof_size=3 \
    --elites_number=5 \
    --precision=float16 \
    --atari_game=pong_v3 \
    --algorithm=GA \
    --max_timesteps_per_episode=1000 \
    --max_evaluation_steps=1000