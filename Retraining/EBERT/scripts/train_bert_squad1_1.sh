#!/bin/bash
export WANDB_PROJECT="EBERT-lab"
export WANDB_DISABLED="true"

RUN_NAME=bert_squad1.1_base_2

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path ./bert-base-uncased \
    --data_dir ./squad_v1 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file train-v1.1.json \
    --predict_file dev-v1.1.json \
    --per_gpu_train_batch_size 12 \
    --per_gpu_eval_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps 100 \
    --save_steps 0 \
    --output_dir fn_base/$RUN_NAME \
    --finetuning_task SQuAD1.1 \
    --run_name $RUN_NAME
  