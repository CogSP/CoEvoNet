#!/bin/bash

python3.9 main.py \
    --debug \
    --train \
    --save \
    --generations=2 \
    --population=4 \
    --hof_size=1 \
    --elites_number=2 \
    --precision=float16 \
    --atari_game=pong_v3 \
    --algorithm=GA \
    --max_timesteps_per_episode=1000 \
    --max_evaluation_steps=1000