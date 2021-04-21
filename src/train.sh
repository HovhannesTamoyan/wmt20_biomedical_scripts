#!/bin/bash

unset DATASET
unset SRC_LANG
unset TGT_LANG


LEARNING_RATE="1e-5"
BATCH_SIZE="256"
MAX_TOKENS="3584"
LABEL_SMOOTHING="0.1"
UPSAMPLE_PRIMARY="1"
SAVE_INTERVAL="1"
VALIDATE_INTERVAL="5"
CLIP_NORM="0.0"
DROPOUT="0.0"
WARMUP="4000"
PATIENCE="5"
MODEL_VERSION="single_model"
VALID_SUBSET="valid,"
EVAL_BLEU=("--eval-bleu" "--eval-bleu-detok" "moses" "--eval-bleu-remove-bpe")
PREFIX=""
LOG_DIR=""
UPDATE_FREQ="1"
WHETHER_RESET=("--reset-meters")
CAB=()
SEED_ARGS=()

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -s|--source-lang)
    SRC_LANG="$2"
    shift # past argument
    shift # past value
    ;;
    -t|--target-lang)
    TGT_LANG="$2"
    shift # past argument
    shift # past value
    ;;
    -i|--dataset)
    DATASET="$2"
    shift # past argument
    shift # past value
    ;;
    -bs|--batch-size)
    BATCH_SIZE="$2"
    shift # past argument
    shift # past value
    ;;
    -mt|--max-tokens)
    MAX_TOKENS="$2"
    shift # past argument
    shift # past value
    ;;
    -ls|--label-smoothing)
    LABEL_SMOOTHING="$2"
    shift # past argument
    shift # past value
    ;;
    -up|--upsample-primary)
    UPSAMPLE_PRIMARY="$2"
    shift # past argument
    shift # past value
    ;;
    -si|--save-interval)
    SAVE_INTERVAL="$2"
    shift # past argument
    shift # past value
    ;;
    -vi|--validate-interval)
    VALIDATE_INTERVAL="$2"
    shift # past argument
    shift # past value
    ;;
    -cn|--clip-norm)
    CLIP_NORM="$2"
    shift # past argument
    shift # past value
    ;;
    -d|--dropout)
    DROPOUT="$2"
    shift # past argument
    shift # past value
    ;;
    -w|--warmup)
    WARMUP="$2"
    shift # past argument
    shift # past value
    ;;
    -i|--dataset)
    DATASET="$2"
    shift # past argument
    shift # past value
    ;;
    -lr|--learning-rate)
    LEARNING_RATE="$2"
    shift # past argument
    shift # past value
    ;;
    -pt|--patience)
    PATIENCE="$2"
    shift # past argument
    shift # past value
    ;;
    -uf|--update-freq)
    UPDATE_FREQ="$2"
    PREFIX="${PREFIX}uf${UPDATE_FREQ}-"
    shift # past argument
    shift # past value
    ;;
    --seed)
    SEED_ARGS=("--seed" "$2")
    PREFIX="${PREFIX}seed$2-"
    shift # past argument
    shift # past value
    ;;
    -reset|--reset)
    WHETHER_RESET=("--reset-meters" "--reset-optimizer" "--reset-lr-scheduler")
    PREFIX="${PREFIX}reset-"
    shift # past argument
    ;;
    -ot|--on-test)
    VALID_SUBSET=""
    PREFIX="${PREFIX}ot-"
    shift # past argument
    ;;
    -nb|--no-bleu)
    EVAL_BLEU=""
    shift # past argument
    ;;
    -cab|--custom-adam-betas)
    CAB=("--adam-betas" "(0.9,0.98)")
    PREFIX="${PREFIX}cab-"
    shift # past argument
    ;;
    -log-dir|--log-dir)
    LOG_DIR="/ssd"
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if ! [ -z "$1" ] 
then
    echo "Positional arguments are set! $1 $2"
    exit
fi

echo "SRC_LANG = $SRC_LANG"
if [ -z "$SRC_LANG" ] 
then
    echo "SRC_LANG is not set"
    exit
fi
echo "TGT_LANG = $TGT_LANG"
if [ -z "$TGT_LANG" ] 
then
    echo "TGT_LANG is not set"
    exit
fi
echo "DATASET = $DATASET"
if ! [ -d /datasets/"${DATASET}/bin" ]
then
    echo "${DATASET} is not a valid directory!"
    exit
fi



DIRECTION="${SRC_LANG}2${TGT_LANG}"
PRETRAINED_MODEL="wmt19.${SRC_LANG}-${TGT_LANG}"

([[ $DIRECTION = "en2de" ]] || [[ $DIRECTION = "de2en" ]]) \
    && PRETRAINED_MODEL="${PRETRAINED_MODEL}.joined-dict"
([[ $DIRECTION = "en2de" ]] || [[ $DIRECTION = "de2en" ]]) \
    && ARCH_CONFIG="--share-all-embeddings" \
    || ARCH_CONFIG=""

PRETRAINED_MODEL="${PRETRAINED_MODEL}.${MODEL_VERSION}"

if [[ $DIRECTION = "en2de" ]] || [[ $DIRECTION = "de2en" ]]; then
    TEST_SUBSET="/datasets/wmt19.biomed.de-en.medline_test/test"
fi
if [[ $DIRECTION = "en2ru" ]] || [[ $DIRECTION = "ru2en" ]]; then
    TEST_SUBSET="/datasets/wmt20.biomed.ru-en.medline_pretest/test"
fi
VALID_SUBSET="${VALID_SUBSET}${TEST_SUBSET}"

MODEL_NAME="${DIRECTION}/${DATASET}/H/${PREFIX}d${DROPOUT}-lr${LEARNING_RATE}-ls${LABEL_SMOOTHING}-w${WARMUP}-bs${BATCH_SIZE}-mt${MAX_TOKENS}-up${UPSAMPLE_PRIMARY}-cn${CLIP_NORM}-vi${VALIDATE_INTERVAL}"
export OUTPUT_DIR="${LOG_DIR}/logs/${MODEL_NAME}"
SAVE_DIR="${OUTPUT_DIR}/save/"
TENSORBOARD_DIR="${OUTPUT_DIR}/"
PRETRAINED_CHECKPOINT="/pretrained/${PRETRAINED_MODEL}/model.pt"
LAST_CHECKPOINT="${SAVE_DIR}/checkpoint_last.pt"
BEST_CHECKPOINT="${SAVE_DIR}/checkpoint_best.pt"
if [ -f $LAST_CHECKPOINT ]; then
    RESTORE_FILE="${LAST_CHECKPOINT}"
    WHETHER_RESET=()
elif [ -d $SAVE_DIR ]; then
    echo "no last checkpoint is detected... in $SAVE_DIR"
    exit
else
    RESTORE_FILE="${PRETRAINED_CHECKPOINT}"
fi

SHARDS=0
for DIR in /datasets/${DATASET}/bin*; do
    ((SHARDS++))
done
MAX_SHARD_ID=$(( $SHARDS-1 ))

DATASET_PATHS="/datasets/${DATASET}/bin"
for IDX in `seq 1 1 ${MAX_SHARD_ID}`; do
    DATASET_PATHS="${DATASET_PATHS}:/datasets/${DATASET}/bin${IDX}"
done
echo $DATASET_PATHS

echo "Writing to ${MODEL_NAME}"

fairseq-train \
  --task "translation" --source-lang "${SRC_LANG}" --target-lang "${TGT_LANG}" \
  --arch "transformer_wmt_en_de_big" $ARCH_CONFIG \
  --encoder-ffn-embed-dim "8192" --share-decoder-input-output-embed \
  --dropout "${DROPOUT}" --relu-dropout "${DROPOUT}" \
  --optimizer "adam" --lr $LEARNING_RATE --lr-scheduler "inverse_sqrt" --warmup-updates "${WARMUP}" \
  --clip-norm "${CLIP_NORM}" \
  --batch-size "${BATCH_SIZE}" --max-tokens "${MAX_TOKENS}" \
  --criterion "label_smoothed_cross_entropy" --label-smoothing "${LABEL_SMOOTHING}" \
  --data-buffer-size "0" --distributed-no-spawn --fp16  \
  --save-interval "${SAVE_INTERVAL}" \
  --validate-interval "${VALIDATE_INTERVAL}" ${EVAL_BLEU[@]} \
  --upsample-primary "${UPSAMPLE_PRIMARY}"\
  --no-epoch-checkpoints --save-interval "${VALIDATE_INTERVAL}" --patience "${PATIENCE}" \
  "${DATASET_PATHS}" \
  --save-dir              "${SAVE_DIR}"  \
  --tensorboard-logdir    "${TENSORBOARD_DIR}" \
  --valid-subset "${VALID_SUBSET}" \
  --restore-file "${RESTORE_FILE}" \
  ${WHETHER_RESET[@]} ${CAB[@]} ${SEED_ARGS[@]} --update-freq "${UPDATE_FREQ}"
