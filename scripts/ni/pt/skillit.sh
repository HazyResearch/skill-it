#!/bin/bash
SEED=0

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir /dfs/scratch0/mfchen/natural-instructions \
	    --val_data_dir /dfs/scratch0/mfchen/natural-instructions \
        --selection_seed ${SEED} \
        --max_steps 5000 \
        --k 23 \
        --slice_list ./aux_data/ni_subsets/top_23.txt \
        --sample_rule mixture \
        --graph_path  ./ni_graphs/top_23_early_decreases.npy \
        --eta 0.2 \
        --num_ckpts 10 \
        --filter_val_skills
done 