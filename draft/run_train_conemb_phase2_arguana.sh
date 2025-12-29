# ================================================================
# Phase 2: Stance Learning
# Goal: Train Stance Head using stance data (InfoNCE + QR Loss)
# Freeze Backbone + Content Head
# ================================================================

echo "Starting Phase 2: Stance Learning"

export CUDA_VISIBLE_DEVICES=2

python train.py \
    --model_name our_bert_cl \
    --model_name_or_path results/our-bge-arguana-qr-phase1 \
    --train_file data/arguana_training_final.csv \
    --eval_file data/arguana_validation_final.csv \
    --output_dir results/our-bge-arguana-qr-phase2 \
    --gradient_checkpointing True \
    --num_train_epochs 20 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --pad_to_max_length True \
    --pooler_type avg \
    --overwrite_output_dir \
    --temp 0.02 \
    --overwrite_cache True \
    --dataloader_drop_last False \
    --do_train \
    --do_eval \
    --fp16 \
    \
    --phase 2 \
    --content_head_dim 768 \
    --stance_head_dim 128 \
    --content_loss_weight 0.0 \
    --stance_loss_weight 1.0 \
    --qr_loss_weight 1.0 \
    "$@"

echo "Phase 2 Completed. Final model saved to results/our-bge-arguana-qr-phase2"
