#!/bin/bash
for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name lego \
        --selection_seed ${SEED} \
        --mw \
        --n_select 0 \
        --max_steps 6000 \
        --batch_size 32 \
        --context_length 128 \
        --k 5 \
        --update_steps 1000 \
        --sample_rule mixture \
        --mw_window 3 \
        --eta 0.5 \
        --num_ckpts 120 \
        --graph 1 0 0 0 0 1 1 1 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1
done 