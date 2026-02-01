#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1

MODEL_DIR=results/decouple/arguana/bge/gated_mlp/lr5e-5_ep3_fbFalse_temp0.02/20260201-191855

echo "$MODEL_DIR"

MODEL="bge"
DATASET="arguana"
TEST_TYPE="test"
QUERY_TRANSFORM_ON=False
QUERY_TRANSFORM_TYPE="gated_mlp"

OUT_DIR=${MODEL_DIR}/${TEST_TYPE}/${DATASET}/use_query_transform_${QUERY_TRANSFORM_ON}

mkdir -p ${OUT_DIR}

python -u test.py \
  --dataset_name ${DATASET} \
  --split ${TEST_TYPE} \
  --model_name_or_path ${MODEL_DIR}/model \
  --model_name ${MODEL} \
  --write_path ${OUT_DIR} \
  --pooler_type avg \
  --max_seq_length 512 \
  --batch_size 64 \
  --k_neighbors 1000 \
  --use_query_transform ${QUERY_TRANSFORM_ON} \
  --query_transform_scale 1.0 \
  --query_transform_type ${QUERY_TRANSFORM_TYPE} \
  2>&1 | tee ${OUT_DIR}/${TEST_TYPE}_eval.txt

cp ${OUT_DIR}/${TEST_TYPE}_eval.txt \
   ${MODEL_DIR}/${TEST_TYPE}_eval.txt