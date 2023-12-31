#!/bin/bash 
export WANDB_PROJECT="EBERT-lab"
export WANDB_DISABLED="true"

TASK_NAME=SST-2
RUN_NAME=bert_sst2_large

CUDA_VISIBLE_DEVICES=1 python run_glue.py \
    --model_name_or_path ./bert-large-uncased \
    --task_name $TASK_NAME \
    --data_dir ./glue_data/$TASK_NAME \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --max_seq_length 64 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --num_train_epochs 3.0 \
    --save_steps 2000  \
    --save_total_limit 1  \
    --output_dir ./fn_large/$TASK_NAME/$RUN_NAME \
    --logging_dir ./fn_large/$TASK_NAME/$RUN_NAME \
    --logging_steps 50 \
    --run_name $RUN_NAME \
    --disable_tqdm 
