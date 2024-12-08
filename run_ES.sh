#!/bin/bash

python3.9 main.py \
    --algorithm=ES \
    --train \
    --save \
    --population=100 \
    --generations=1000000 \
    --hof_size=1 \
    --learning_rate=0.1 \
    --precision=float16 \
    --atari_game=pong_v3 \
    --max_timesteps_per_episode=1000 \
    --max_evaluation_steps=1000