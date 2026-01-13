#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=6

BGE=BAAI/bge-base-en-v1.5
UAE=WhereIsAI/UAE-Large-V1
GTE=Alibaba-NLP/gte-large-en-v1.5

MODEL=${BGE}

MODEL_DIR=results/arguana/${MODEL}/queryT_linear/best/finetune/lr5e-5_ep3_fbFalse_hnw0.5_temp0.02/20251230-180509

TS=$(date +"%Y%m%d-%H%M%S")
OUT_DIR=test_results/arguana_cos_only/test

mkdir -p ${OUT_DIR}

python test_contradiction_faiss_final.py \
  --dataset_name arguana \
  --split test \
  --model_name_or_path ${MODEL_DIR}/model \
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