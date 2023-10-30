#!/bin/bash
# skill-it
MAX_STEPS=6000
BATCH_SIZE=32
UPDATE_STEPS=1200


for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name addition \
        --selection_seed ${SEED} \
        --mw \
        --n_select 0 \
        --sample_rule mixture \
        --max_steps ${MAX_STEPS} \
        --batch_size ${BATCH_SIZE} \
        --context_length 128 \
        --k 3 \
        --update_steps ${UPDATE_STEPS} \
        --mw_window 3 \
        --eta 0.1 \
        --num_ckpts 120 \
        --graph 1 0 0 1 1 0 0 0 1
done 
