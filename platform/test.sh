#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=6

MODEL_DIR=results/hotpotqa/uae/queryT_linear/lr5e-5_ep10_fbTrue_hnw0.0_temp0.02/20260117-193015

echo "$MODEL_DIR"

MODEL="uae"
TEST_TYPE="test"
DATASET="hotpotqa"

TS=$(date +"%Y%m%d-%H%M%S")
OUT_DIR=test_results/${DATASET}_cos_only/${TEST_TYPE}/${TS}

mkdir -p ${OUT_DIR}

python test_contradiction_faiss_final.py \
  --dataset_name ${DATASET} \
  --split ${TEST_TYPE} \
  --model_name_or_path ${MODEL_DIR}/model \
  --model_name ${MODEL} \
  --write_path ${OUT_DIR} \
  --pooler_type avg \
  --max_seq_length 512 \
  --batch_size 64 \
  --k_neighbors 1000 \
  --use_query_transform True \
  --query_transform_scale 1.0 \
  2>&1 | tee ${OUT_DIR}/${TEST_TYPE}_eval.txt

cp ${OUT_DIR}/${TEST_TYPE}_eval.txt \
   ${MODEL_DIR}/${TEST_TYPE}_eval.txt