#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

MODEL_DIR=results/arguana/bge/queryT_linear/lr5e-5_ep3_fbFalse_hnw0.0_temp0.01_qtd0.1_qts1.0_qtis0.02/20260105-202746

TS=$(date +"%Y%m%d-%H%M%S")
OUT_DIR=test_results/arguana_cos_only/test/${TS}

mkdir -p ${OUT_DIR}

python test_contradiction_faiss_final.py \
  --dataset_name arguana \
  --split test \
  --model_name_or_path ${MODEL_DIR}/model \
  --model_name bge \
  --write_path ${OUT_DIR} \
  --pooler_type avg \
  --max_seq_length 512 \
  --batch_size 64 \
  --k_neighbors 1000 \
  --use_query_transform True \
  --query_transform_scale 1.0 \
  2>&1 | tee ${OUT_DIR}/test_eval.txt

cp ${OUT_DIR}/test_eval.txt \
   ${MODEL_DIR}/test_eval.txt