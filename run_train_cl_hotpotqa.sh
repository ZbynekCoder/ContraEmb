#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=5

LR=5e-5
EP=3
FB=True
HNW=0.5
TEMP=0.02
QTD=0.1
QTS=1.0
QTIS=0.02

TS=$(date +"%Y%m%d-%H%M%S")
OUT_DIR=results/hotpotqa/queryT_linear/lr${LR}_ep${EP}_fb${FB}_hnw${HNW}_temp${TEMP}/${TS}

mkdir -p ${OUT_DIR}
cp "$0" "${OUT_DIR}/run_train.sh"
cat <<EOF > ${OUT_DIR}/config.txt
Task: HotpotQA contradiction retrieval
Date: $(date)

LR=${LR}
EP=${EP}
FreezeBackbone=${FB}
HardNegWeight=${HNW}
Temp=${TEMP}

QueryTransform:
  dropout=${QTD}
  scale=${QTS}
  init_std=${QTIS}
EOF

python train.py \
  --model_name our_gte \
  --model_name_or_path Alibaba-NLP/gte-large-en-v1.5 \
  --train_file data/hotpotqa_train_gpt4_final.csv \
  --eval_file data/hotpotqa_dev_gpt4_final.csv \
  --output_dir ${OUT_DIR}/model \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --pad_to_max_length True \
  --pooler_type avg \
  --loss_type cos \
  --temp ${TEMP} \
  --hard_negative_weight ${HNW} \
  --num_train_epochs ${EP} \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing True \
  --learning_rate ${LR} \
  --dataloader_drop_last True \
  --fp16 \
  --logging_steps 50 \
  --save_strategy no \
  --evaluation_strategy no \
  --use_query_transform True \
  --freeze_backbone ${FB} \
  --query_transform_dropout ${QTD} \
  --query_transform_scale ${QTS} \
  --query_transform_init_std ${QTIS} \
  2>&1 | tee ${OUT_DIR}/train.log
