#!/bin/bash
MAIN=path/to/main

export CUDA_VISIBLE_DEVICES=0

fairseq-train ${MAIN}/data/preprocess/ \
    --task translation \
    --arch transformer \
    --source-lang ja --target-lang en \
    --max-epoch 10 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --batch-size 32 \
    --update-freq 2 \
    --dropout 0.2 --weight-decay 1e-4 \
    --optimizer adam --clip-norm 1.0 --adam-betas '(0.9, 0.98)' \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 10000 --save-dir model91/ \
    --fp16 > ${MAIN}/logs/train.log
