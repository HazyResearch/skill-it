#!/bin/bash
SEED=0

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir /dfs/scratch0/mfchen/natural-instructions \
	    --val_data_dir /dfs/scratch0/mfchen/natural-instructions \
        --ni_test \
        --selection_seed ${SEED} \
        --max_steps 5000 \
        --k 59 \
        --lr 5e-6 \
        --num_ckpts 10
done 
