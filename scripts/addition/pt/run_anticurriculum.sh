#!/bin/bash
USTEPS=1_200
for SEED in 0 1 2 3 4
do
    P1=13
    P2=14
    P3=18
    python3 main.py \
        --task_name addition \
        --selection_seed ${SEED} \
        --n_select 192000 \
        --max_steps 6000 \
        --update_steps 2000 \
        --batch_size 32 \
        --context_length 128 \
        --sample_rule mixture \
        --k 3 \
        --proportions $P1 $P2 $P3 \
        --num_ckpts 120 \
        --anticurriculum
done

