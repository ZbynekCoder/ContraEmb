#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1
export DEBUG_NAN=1

LR=4e-5
EP=1
BATCH_SIZE=128
FB=False
QT_DROPOUT=0.1
QT_SCALE=1.0
QTIS=0.02
QT_TYPE="linear"

LOSS_TYPE="cos"

MODEL="bge"
DATASET="arguana"
DATASET_TYPE="isolate"

case "$MODEL" in
  bge)
    MODEL_DIR="BAAI/bge-base-en-v1.5"
    TEMP=0.02
    ;;
  uae)
    MODEL_DIR="WhereIsAI/UAE-Large-V1"
    TEMP=0.02
    ;;
  gte)
    MODEL_DIR="Alibaba-NLP/gte-large-en-v1.5"
    TEMP=0.01
    ;;
  *)
    echo "Unknown MODEL: $MODEL" >&2
    exit 1
    ;;
esac
echo "$MODEL_DIR"

case "$DATASET" in
  arguana)
    case "$DATASET_TYPE" in
      isolate)
        TRAINING_DATA="data/arguana_training_final.csv"
        EVAL_DATA="data/arguana_validation_final.csv"
        ;;
      aggregate)
        TRAINING_DATA="data/padded/arguana_training_aggregate.csv"
        EVAL_DATA="data/padded/arguana_validation_aggregate.csv"
        ;;
    esac
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
OUT_DIR=results/${LOSS_TYPE}/${DATASET}/${DATASET_TYPE}/${MODEL}/${QT_TYPE}/lr${LR}_ep${EP}_fb${FB}_temp${TEMP}_bs${BATCH_SIZE}_scale${QT_SCALE}_dropout${QT_DROPOUT}/${TS}

mkdir -p ${OUT_DIR}
cp "$0" "${OUT_DIR}/run_train.sh"
cat <<EOF > ${OUT_DIR}/config.txt
Task: ${DATASET} contradiction retrieval
Date: $(date)

LR=${LR}
EP=${EP}
FreezeBackbone=${FB}
Temp=${TEMP}

QueryTransform:
  dropout=${QTD}
  scale=${QTS}
  init_std=${QTIS}
EOF

python -u train.py \
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
  --loss_type ${LOSS_TYPE} \
  --temp ${TEMP} \
  --num_train_epochs ${EP} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
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
  --query_transform_dropout ${QT_DROPOUT} \
  --query_transform_scale ${QT_SCALE} \
  --query_transform_init_std ${QTIS} \
  --query_transform_type ${QT_TYPE} \
  2>&1 | tee ${OUT_DIR}/train.log