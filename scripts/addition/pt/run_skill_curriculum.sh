#!/bin/bash

USTEPS=1_200
for SEED in 0 1 2 3 4
do
    python main.py \
        --task_name addition \
        --selection_seed ${SEED} \
        --n_select 192000 \
        --max_steps 6000 \
        --batch_size 32 \
        --sample_rule stratified \
        --context_length 128 \
        --k 3 \
        --num_ckpts 120 \
        --group_curriculum \
        --curriculum \
        --mixing_frac 0.0
done