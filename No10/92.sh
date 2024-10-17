#!/bin/bash
MAIN=path/to/main

export CUDA_VISIBLE_DEVICES=0

fairseq-interactive --path ${MAIN}/model91/checkpoint_best.pt ${MAIN}/data/preprocess < ${MAIN}/data/preprocess/test.ja | grep '^H' | cut -f3 > ${MAIN}/data/preprocess/eval.out
