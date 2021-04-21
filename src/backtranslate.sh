#!/bin/bash

unset SRC_LANG
unset TGT_LANG
unset MODEL_DIR
unset DATA_DIR

PREFIX=""

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
    -m|--model)
    MODEL_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    -i|--data_dir)
    DATA_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    -p|--prefix)
    PREFIX="$2"
    shift # past argument
    shift # past value
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
    echo "Positional arguments are set!"
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
echo "MODEL_DIR = $MODEL_DIR"
if ! [ -d $MODEL_DIR ]
then
    echo "${MODEL_DIR} is not a valid directory!"
    exit
fi
echo "DATA_DIR = $DATA_DIR"
if ! [ -d $DATA_DIR ]
then
    echo "${DATA_DIR} is not a valid directory!"
    exit
fi

if ! [ -z "$1" ] 
then
    echo "Unexpected arguments are given!"
    exit
fi

echo -n "Press any key to continue!";
# read;


MODEL_VERSION="single_model"
DIRECTION="${SRC_LANG}2${TGT_LANG}"

PRETRAINED_MODEL="wmt19.${SRC_LANG}-${TGT_LANG}"
([[ $DIRECTION = "en2de" ]] || [[ $DIRECTION = "de2en" ]]) \
    && PRETRAINED_MODEL="${PRETRAINED_MODEL}.joined-dict"
PRETRAINED_MODEL="${PRETRAINED_MODEL}.${MODEL_VERSION}"
PRETRAINED_MODEL_DIR="/pretrained/${PRETRAINED_MODEL}"

BEAM_SIZE="1"
BUFFER_SIZE="4096"
MAX_TOKENS="65536"

BEST_CHECKPOINT="${MODEL_DIR}/save/checkpoint_best.pt"
DEFAULT_CHECKPOINT="${MODEL_DIR}/model.pt"

[[ -f $BEST_CHECKPOINT ]] \
    && CHECKPOINT_PATH="${BEST_CHECKPOINT}" \
    || CHECKPOINT_PATH="${DEFAULT_CHECKPOINT}"

OUT_DIR="${DATA_DIR}/${PREFIX}${DIRECTION}"
mkdir "${OUT_DIR}"
SRC_OUT_PATH="${OUT_DIR}/train.${SRC_LANG}-${TGT_LANG}.${SRC_LANG}"
TGT_OUT_PATH="${OUT_DIR}/train.${SRC_LANG}-${TGT_LANG}.${TGT_LANG}"
LOG_OUT_PATH="${OUT_DIR}/translation.log"
cat "${DATA_DIR}/train.${SRC_LANG}" > "${SRC_OUT_PATH}"

pv "${SRC_OUT_PATH}" \
| fairseq-interactive \
    --input /dev/stdin \
    --path "${CHECKPOINT_PATH}" "${PRETRAINED_MODEL_DIR}" \
    --beam "${BEAM_SIZE}" --source-lang "${SRC_LANG}" --target-lang "${TGT_LANG}" \
    --normalizer moses --tokenizer moses --bpe fastbpe --bpe-codes "${PRETRAINED_MODEL_DIR}/bpecodes.${SRC_LANG}" \
    --fp16 --buffer-size "${BUFFER_SIZE}" --max-tokens "${MAX_TOKENS}" \
    --sampling --sampling-topk 10 --truncate-invalid-size-inputs-valid-test \
| tee "${LOG_OUT_PATH}" \
| grep ^D \
| cut -f3 \
| sed "s/<unk>/֎/g" \
> "${TGT_OUT_PATH}"
