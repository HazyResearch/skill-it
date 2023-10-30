#!/bin/bash
SEED=0

for SEED in 3 4
do 
    python3 main.py \
        --task_name lego \
        --selection_seed ${SEED} \
        --n_select 192000 \
        --max_steps 6000 \
        --batch_size 32 \
        --context_length 128 \
        --sample_rule stratified \
        --k 5 \
        --num_ckpts 120 \
        --group_curriculum \
        --curriculum \
        --mixing_frac 0.0
done 