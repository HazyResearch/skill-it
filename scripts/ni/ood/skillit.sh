#!/bin/bash
SEED=0

for SEED in 0
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir /dfs/scratch0/mfchen/natural-instructions \
	    --val_data_dir /dfs/scratch0/mfchen/natural-instructions \
        --ni_test \
        --selection_seed ${SEED} \
        --max_steps 5000 \
        --k 59 \
        --sample_rule mixture \
        --mw \
        --update_steps 500 \
        --mw_window 3 \
        --eta 0.2 \
        --graph_path ./ni_graphs/ni_test_graph.npy \
        --lr 5e-6 \
        --num_ckpts 10
done 
