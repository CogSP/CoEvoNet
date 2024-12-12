#!/bin/bash

python3.9 main.py \
    --algorithm=GA \
    --test \
    --ES_model_to_test="ES_models/gens500_pop200_hof1_gamesimple_adversary_v3_mut0.05_adaptiveFalse_lr0.1_tslimit400/agent.pth" \
    --game="simple_adversary_v3" \
    --render \