#!/bin/bash
for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name lego \
        --selection_seed ${SEED} \
        --n_select 192000 \
        --batch_size 32 \
        --context_length 128 \
        --k 5 \
        --sample_rule mixture \
        --proportions 1 1 1 0 0
done 