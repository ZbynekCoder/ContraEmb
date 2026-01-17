#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=4

MODEL_DIR=results/msmarco/BAAI/bge-base-en-v1.5/queryT_linear/lr5e-5_ep1_fbTrue_hnw0.0_temp0.02/20260106-223856

TS=$(date +"%Y%m%d-%H%M%S")
OUT_DIR=test_results/msmarco_cos_only/dev/${TS}

mkdir -p ${OUT_DIR}

python test_contradiction_faiss_final.py \
  --dataset_name msmarco \
  --split dev \
  --model_name_or_path ${MODEL_DIR}/model \
  --write_path ${OUT_DIR} \
  --pooler_type avg \
  --max_seq_length 512 \
  --batch_size 64 \
  --k_neighbors 1000 \
  --use_query_transform True \
  --query_transform_scale 1.0 \
  2>&1 | tee ${OUT_DIR}/dev_eval.txt

cp ${OUT_DIR}/dev_eval.txt \
   ${MODEL_DIR}/dev_eval.txt