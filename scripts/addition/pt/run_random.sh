#!/bin/bash
MAX_STEPS=6000
BATCH_SIZE=32
N_SELECT=$(($MAX_STEPS*$BATCH_SIZE))
UPDATE_STEPS=1200


for SEED in 2 3 4
do 
    # random sampling
    P1=13
    P2=14
    P3=18
    python3 main.py \
        --task_name addition \
        --selection_seed ${SEED} \
        --n_select ${N_SELECT} \
        --batch_size ${BATCH_SIZE} \
        --context_length 128 \
        --sample_rule mixture \
        --k 3 \
        --proportions $P1 $P2 $P3 \
        --num_ckpts 60
done 
