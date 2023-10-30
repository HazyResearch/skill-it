#!/bin/bash
# sample law script.
N_SELECT=10_000
for SEED in 0 1 2 3 4
do 
    # select law points 
    python3 main.py \
        --task_name law \
        --train_data_dir /dfs/scratch0/mfchen/data/smallest_law_and_other/train \
	--val_data_dir /dfs/scratch0/mfchen/data/smallest_law_val \
        --selection_seed ${SEED} \
        --n_select ${N_SELECT} \
        --sample_rule random \
        --slice_list ./aux_data/law_slices.txt
done 
