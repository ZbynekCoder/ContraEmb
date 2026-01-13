#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=7

LR=5e-5
EP=3
TEMP=0.02
HNW=0.0
QTD=0.1
QTS=1.0
QTIS=0.02

BGE=BAAI/bge-base-en-v1.5
UAE=WhereIsAI/UAE-Large-V1
GTE=Alibaba-NLP/gte-large-en-v1.5

MODEL=BGE

TS=$(date +"%Y%m%d-%H%M%S")
OUT_DIR=results/arguana/${MODEL}/dual_tower/lr${LR}_ep${EP}_temp${TEMP}_hnw${HNW}/${TS}

mkdir -p ${OUT_DIR}
cp "$0" "${OUT_DIR}/run_train.sh"
cat <<EOF > ${OUT_DIR}/config.txt
Task: ArguAna contradiction retrieval
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
  --model_name our_model \
  --model_name_or_path ${MODEL} \
  --doc_model_name_or_path ${MODEL} \
  --use_dual_encoder True \
  --freeze_doc_encoder True \
  --train_file data/arguana_training_final.csv \
  --eval_file data/arguana_validation_final.csv \
  --output_dir ${OUT_DIR}/model \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
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
  --use_query_transform False \
  2>&1 | tee ${OUT_DIR}/train.log
