#!/bin/bash

python3.9 main.py \
    --algorithm=ES \
    --test \
    --ES_model_to_test_agent_0="ES_models/gens500_pop30_hof1_gamesimple_adversary_v3_tslimit400_fitness-sharingTrue_adaptiveTruemax_mutation0.5_min_mutation0.001_lr0.1/agent_0.pth" \
    --ES_model_to_test_agent_1="ES_models/gens500_pop30_hof1_gamesimple_adversary_v3_tslimit400_fitness-sharingTrue_adaptiveTruemax_mutation0.5_min_mutation0.001_lr0.1/agent_1.pth" \
    --ES_model_to_test_adversary_0="ES_models/gens500_pop30_hof1_gamesimple_adversary_v3_tslimit400_fitness-sharingTrue_adaptiveTruemax_mutation0.5_min_mutation0.001_lr0.1/adversary.pth" \
    --game="simple_adversary_v3" \
    --render \