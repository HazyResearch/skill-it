#!/bin/bash
SEED=0

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name alpaca \
        --train_data_dir ./aux_data/alpaca_final.pkl \
	    --val_data_dir ./alpaca_final.pkl \
        --dev_split_path ./aux_data/alpaca_dev_split_map.pkl \
        --selection_seed ${SEED} \
        --n_select 4000 \
        --num_ckpts 10 \
        --sample_rule stratified
done
