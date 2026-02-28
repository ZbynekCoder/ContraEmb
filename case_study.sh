#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=5

MODEL="bge"
TEST_TYPE="test"
DATASET="arguana"
QUERY_TRANSFORM_ON=True
QUERY_TRANSFORM_TYPE="linear"
CASE_STRATEGY="T_wins"

python case_study.py \
  --dataset_name ${DATASET} \
  --split ${TEST_TYPE} \
  --finetuned_model_name_or_path results/cos/arguana/isolate/bge/linear/lr3e-5_ep1_fbFalse_temp0.02_bs128_scale1.0_dropout0.1/20260225-185733/model \
  --baseline_model_name BAAI/bge-base-en-v1.5 \
  --write_path ./case_study \
  --index_dir ./indices_case \
  --k_neighbors 1000 \
	--use_query_transform ${QUERY_TRANSFORM_ON} \
  --query_transform_scale 1.0 \
  --query_transform_type ${QUERY_TRANSFORM_TYPE} \
  --case_topk 10 \
  --case_max_q 30 \
	--case_out case_study_${CASE_STRATEGY}.jsonl \
  --case_strategy ${CASE_STRATEGY}
