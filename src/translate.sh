#!/bin/bash

unset SRC_LANG
unset TGT_LANG
unset MODEL_DIR

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -s|--source-lang)   SRC_LANG="$2"       && shift && shift ;;
        -t|--target-lang)   TGT_LANG="$2"       && shift && shift ;;
        -m|--model)         MODEL_DIR="$2"      && shift && shift ;;
        *)                  POSITIONAL+=("$1")  && shift ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

METHOD=$1
VAL=$2


CKPT="best"
if ! [ -z "$3" ] ; then
    CKPT=$3
fi

if ! [ -z "$4" ] ; then
    echo "Extra positional arguments are set! ($4)"; exit
fi

if [ -z "$SRC_LANG" ] ; then
    echo "SRC_LANG is not set"; exit
fi

if [ -z "$TGT_LANG" ] ; then
    echo "TGT_LANG is not set"; exit
fi

if ! [ -d $MODEL_DIR ] ; then
    echo "${MODEL_DIR} is not a valid directory!"; exit
fi


MODEL_VERSION="single_model"
DIRECTION="${SRC_LANG}2${TGT_LANG}"

PRETRAINED_MODEL="wmt19.${SRC_LANG}-${TGT_LANG}"
([[ $DIRECTION = "en2de" ]] || [[ $DIRECTION = "de2en" ]]) \
    && PRETRAINED_MODEL="${PRETRAINED_MODEL}.joined-dict"
PRETRAINED_MODEL="${PRETRAINED_MODEL}.${MODEL_VERSION}"
PRETRAINED_MODEL_DIR="/pretrained/${PRETRAINED_MODEL}"

BUFFER_SIZE="4096"
MAX_TOKENS="16384"

BEST_CHECKPOINT="${MODEL_DIR}/save/checkpoint_${CKPT}.pt"
DEFAULT_CHECKPOINT="${MODEL_DIR}/model.pt"

[[ -f $BEST_CHECKPOINT ]] \
    && CHECKPOINT_PATH="${BEST_CHECKPOINT}" \
    || CHECKPOINT_PATH="${DEFAULT_CHECKPOINT}"



if [[ $METHOD == "greedy" ]]; then
    BEAM_SIZE="1"
    EXTRA_ARGS=("--beam" "${BEAM_SIZE}")
elif [[ $METHOD == "beam" ]]; then
    BEAM_SIZE="${VAL}"
    EXTRA_ARGS=("--beam" "${BEAM_SIZE}")
elif [[ $METHOD == "sampling" ]]; then
    BEAM_SIZE=1
    TOPK="${VAL}"
    if [[ -z TOPK ]]; then
        EXTRA_ARGS=("--beam" "${BEAM_SIZE}" "--sampling")
    else
        EXTRA_ARGS=("--beam" "${BEAM_SIZE}" "--sampling" "--sampling-topk" $TOPK)
    fi
else
    echo "Method ${METHOD} is not implemented!" >&2 ; exit
fi

MAX_TOKENS=$(( $MAX_TOKENS / $BEAM_SIZE ))

FPFLAGS="--fp16"
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "No GPUs detected..." >&2
    echo "Falling back to CPU mode (FLOAT32, MAX_TOKENS=4096)" >&2
    FPFLAGS=""
    MAX_TOKENS="4096"
    BUFFER_SIZE="128"
fi

fairseq-interactive --input /dev/stdin \
  --path "${CHECKPOINT_PATH}" "${PRETRAINED_MODEL_DIR}" \
  --source-lang "${SRC_LANG}" --target-lang "${TGT_LANG}" \
  --normalizer moses \
    --moses-replace-unicode-puncts \
    --moses-normalize-quote-commas \
    --moses-normalize-numbers \
    --moses-remove-control-chars \
  --tokenizer moses \
  --bpe fastbpe --bpe-codes "${PRETRAINED_MODEL_DIR}/bpecodes.${SRC_LANG}" \
  $FPFLAGS --buffer-size "${BUFFER_SIZE}" --max-tokens "${MAX_TOKENS}" \
  --truncate-invalid-size-inputs-valid-test \
  ${EXTRA_ARGS[@]} \
| tee last.translation.log \
| grep ^D \
| cut -f3 \
| sed "s/<unk>/֎/g"
