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
        --k 2 \
        --sample_rule mixture \
        --target_mask 1 0 \
        --mw \
        --eta 0.5 \
        --mw_window 3 \
        --update_steps 100 \
        --graph_path ./ni_graphs/stance_detection.npy \
        --filter_val_skills \
        --num_ckpts 6 \
        --session_id stance_detection_debug 
done 
