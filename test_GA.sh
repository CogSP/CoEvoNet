#!/bin/bash

python3.9 main.py \
    --algorithm=GA \
    --test \
    --GA_hof_to_test_agent_0="GA_models/gens500_pop20_hof3_gamesimple_adversary_v3_tslimit400_fitness-sharingTrue_adaptiveTruemax_mutation0.7_min_mutation0.0001/hall_of_fame_agent_0.pth" \
    --GA_hof_to_test_agent_1="GA_models/gens500_pop20_hof3_gamesimple_adversary_v3_tslimit400_fitness-sharingTrue_adaptiveTruemax_mutation0.7_min_mutation0.0001/hall_of_fame_agent_1.pth" \
    --GA_hof_to_test_adversary="GA_models/gens500_pop20_hof3_gamesimple_adversary_v3_tslimit400_fitness-sharingTrue_adaptiveTruemax_mutation0.7_min_mutation0.0001/hall_of_fame_adversary.pth" \
    --game="simple_adversary_v3" \
    --render \