#!/bin/bash

# ================================================================
# Phase 1: Content Pre-training
# Goal: Train Backbone + Content Head using content similarity (InfoNCE)
# ================================================================

echo "Starting Phase 1: Content Pre-training"

export CUDA_VISIBLE_DEVICES=4

python train.py \
    --model_name our_bge_qr_phase1 \
    --model_name_or_path BAAI/bge-base-en-v1.5 \
    --train_file data/arguana_training_final.csv \
    --eval_file data/arguana_validation_final.csv \
    --output_dir results/our-bge-arguana-qr-phase1 \
    --gradient_checkpointing True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --max_seq_length 512 \
    --pad_to_max_length True \
    --pooler_type avg \
    --overwrite_output_dir \
    --temp 0.02 \
    --dataloader_drop_last True \
    --do_train \
    --do_eval \
    --fp16 \
    \
    --phase 1 \
    --content_head_dim 768 \
    --stance_head_dim 128 \
    --content_loss_weight 1.0 \
    --stance_loss_weight 0.0 \
    --qr_loss_weight 0.0 \
    "$@"

echo "Phase 1 Completed. Model saved to results/our-bge-arguana-qr-phase1"
