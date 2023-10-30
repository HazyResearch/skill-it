#!/bin/bash
SEED=0

for SEED in 3 4
do 

    P1=1
    P2=1
    P3=1
    P4=3
    P5=5

    python3 main.py \
        --task_name lego \
        --selection_seed ${SEED} \
        --n_select 192000 \
        --max_steps 6000 \
        --batch_size 32 \
        --sample_rule mixture \
        --context_length 128 \
        --update_steps 1000 \
        --k 5 \
        --proportions $P1 $P2 $P3 $P4 $P5 \
        --num_ckpts 120 \
        --curriculum
done 