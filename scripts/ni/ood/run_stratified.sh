#!/bin/bash
for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir /dfs/scratch0/mfchen/natural-instructions \
	    --val_data_dir /dfs/scratch0/mfchen/natural-instructions \
        --ni_test \
        --selection_seed ${SEED} \
        --max_steps 5000 \
        --k 13 \
        --sample_rule stratified \
        --slice_list explanation information_extraction misc. question_answering question_generation question_understanding sentiment_analysis style_transfer summarization text_categorization text_completion text_to_code word_semantics \
        --num_ckpts 10 \
        --lr 5e-6
done 
