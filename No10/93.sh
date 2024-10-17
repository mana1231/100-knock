#!/bin/bash
MAIN=path/to/main

export CUDA_VISIBLE_DEVICES=0

fairseq-score \
    --sys ${MAIN}/data/preprocess/eval.out \
    --ref ${MAIN}/data/preprocess/test.en