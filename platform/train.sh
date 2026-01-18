#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=6

LR=5e-5
EP=10
FB=True
HNW=0.0
TEMP=0.02
QTD=0.1
QTS=1.0
QTIS=0.02

MODEL="gte"
DATASET="hotpotqa"

case "$MODEL" in
  bge)
    MODEL_DIR="BAAI/bge-base-en-v1.5"
    ;;
  uae)
    MODEL_DIR="WhereIsAI/UAE-Large-V1"
    ;;
  gte)
    MODEL_DIR="Alibaba-NLP/gte-large-en-v1.5"
    ;;
  *)
    echo "Unknown MODEL: $MODEL" >&2
    exit 1
    ;;
esac
echo "$MODEL_DIR"

case "$DATASET" in
  arguana)
    TRAINING_DATA="data/arguana_training_final.csv"
    EVAL_DATA="data/arguana_validation_final.csv"
    ;;
  msmarco)
    TRAINING_DATA="data/msmarco_train_gpt4_final.csv"
    EVAL_DATA="data/msmarco_dev_gpt4_final.csv"
    ;;
  hotpotqa)
    TRAINING_DATA="data/hotpotqa_train_gpt4_final.csv"
    EVAL_DATA="data/hotpotqa_dev_gpt4_final.csv"
    ;;
  *)
    echo "Unknown DATASET: $DATASET" >&2
    exit 1
    ;;
esac
echo "$DATASET"

TS=$(date +"%Y%m%d-%H%M%S")
OUT_DIR=results/${DATASET}/${MODEL}/queryT_linear/lr${LR}_ep${EP}_fb${FB}_hnw${HNW}_temp${TEMP}/${TS}

mkdir -p ${OUT_DIR}
cp "$0" "${OUT_DIR}/run_train.sh"
cat <<EOF > ${OUT_DIR}/config.txt
Task: ${DATASET} contradiction retrieval
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
  --model_name our_${MODEL} \
  --model_name_or_path ${MODEL_DIR} \
  --train_file ${TRAINING_DATA} \
  --eval_file ${EVAL_DATA} \
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
