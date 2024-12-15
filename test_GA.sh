#!/bin/bash

python3.9 main.py \
    --algorithm=GA \
    --test \
    --GA_hof_to_test="GA_models/gens500_pop200_hof3_gamesimple_adversary_v3_mut0.005_adaptiveFalse_tslimit400/elite_weights.pth" \
    --game="simple_adversary_v3" \
    --render \