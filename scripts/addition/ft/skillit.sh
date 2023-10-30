#!/bin/bash
MAX_STEPS=6000
BATCH_SIZE=32
UPDATE_STEPS=1200

for SEED in 0 1 2 3 4
do 
    # skill-it
    python3 main.py \
        --task_name addition \
        --selection_seed ${SEED} \
        --max_steps ${MAX_STEPS} \
        --n_select 0 \
        --batch_size ${BATCH_SIZE} \
        --k 3 \
        --context_length 128 \
        --sample_rule mixture \
        --slice_list 0 1 \
        --target_mask 1 0 \
        --mw \
        --eta 0.1 \
        --mw_window 3 \
        --update_steps ${UPDATE_STEPS} \
        --graph 1 0 1 1 \
        --num_ckpts 60 \
        --filter_val_skills
done 