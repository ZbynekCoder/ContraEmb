#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1

TS=$(date +"%Y%m%d-%H%M%S")
TRAIN_DIR=results/arguana/dual_tower/lr5e-5_ep3_temp0.02_hnw0.0/20260106-183042

TS=$(date +"%Y%m%d-%H%M%S")
OUT_DIR=test_results/arguana_dual/dev/${TS}
mkdir -p ${OUT_DIR}

python test_contradiction_faiss_final.py \
  --dataset_name arguana \
  --split dev \
  --model_name_or_path ${TRAIN_DIR}/model \
  --write_path ${OUT_DIR} \
  --pooler_type avg \
  --max_seq_length 512 \
  --batch_size 64 \
  --k_neighbors 1000 \
  2>&1 | tee ${OUT_DIR}/dev_eval.txt
