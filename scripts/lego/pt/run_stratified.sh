#!/bin/bash
for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name lego \
        --selection_seed ${SEED} \
        --n_select 0 \
        --max_steps 6000 \
        --sample_rule stratified \
        --batch_size 32 \
        --context_length 128 \
        --k 5 \
        --num_ckpts 120
done 