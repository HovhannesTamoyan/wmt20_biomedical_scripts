#!/bin/bash

unset SRC_LANG
unset TGT_LANG
unset MODEL_DIR
unset DATA_DIR

VERSION=""

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
    -i|--data-pref)
    DATA_PREF="$2"
    shift # past argument
    shift # past value
    ;;
    -pp|--pipelined)
    VERSION="_pipeline"
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


if [ -z "$SRC_LANG" ] 
then
    echo "SRC_LANG is not set"
    exit
fi

if [ -z "$TGT_LANG" ] 
then
    echo "TGT_LANG is not set"
    exit
fi

if ! [ -d $MODEL_DIR ]
then
    echo "${MODEL_DIR} is not a valid directory!"
    exit
fi

if ! [ -f "$DATA_PREF.$SRC_LANG" ]
then
    echo "${DATA_PREF} is not a valid prefix for input sentences!"
    exit
fi
if ! [ -f "$DATA_PREF.$TGT_LANG" ]
then
    echo "${DATA_PREF} is not a valid prefix for references!"
    exit
fi

echo "${MODEL_DIR}"

cat "${DATA_PREF}.${SRC_LANG}" \
| "./translate${VERSION}.sh" -s "${SRC_LANG}" -t "${TGT_LANG}" -m "${MODEL_DIR}" $1 $2 $3 \
| tee "/tmp/${SRC_LANG}.${TGT_LANG}.$1.$2${VERSION}" \
| pv -t -l -s $(cat "${DATA_PREF}.${SRC_LANG}" | wc -l) -W \
| sacrebleu -l "${SRC_LANG}-${TGT_LANG}" "${DATA_PREF}.${TGT_LANG}"
