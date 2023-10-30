#!/bin/bash
SEED=0

for SEED in 0 1 2 3 4
do 

    python3 main.py \
        --task_name lego \
        --selection_seed ${SEED} \
        --max_steps 6000 \
        --n_select 0 \
        --batch_size 32 \
        --k 5 \
        --context_length 128 \
        --slice_list 0 1 2 \
        --sample_rule mixture \
        --target_mask 0 0 1  \
        --mw \
        --eta 0.5 \
        --mw_window 3 \
        --update_steps 500 \
        --graph 1 0 0 1 1 1 0 0 1 \
        --num_ckpts 120 \
        --filter_val_skills
done 