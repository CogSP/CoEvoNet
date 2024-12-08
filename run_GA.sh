#!/bin/bash

python3.9 main.py \
    --train \
    --save \
    --population=10 \
    --hof_size=10 \
    --elites_number=2 \
    --precision=float16 \
    --atari_game=pong_v3 \
    --algorithm=GA