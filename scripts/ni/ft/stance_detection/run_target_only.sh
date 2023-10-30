#!/bin/bash
SEED=0

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir /dfs/scratch0/mfchen/natural-instructions \
	    --val_data_dir /dfs/scratch0/mfchen/natural-instructions \
        --selection_seed ${SEED} \
        --max_steps 600 \
        --slice_list stance_detection text_matching \
        --sample_rule mixture \
        --proportions 1 0 \
        --num_ckpts 6 \
        --filter_val_skills \
        --session_id stance_detection_debug 
done 
