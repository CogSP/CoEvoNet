#!/bin/bash

python3.9 main.py \
    --algorithm=ES \
    --test \
    --ES_model_to_test=../kaggle_COEV/first_take_gen500_pop200_hofsize1_lr0.1_maxtimesteps1000_pongv3/agent.pth \
    --atari_game=pong_v3 \
    --render \

    