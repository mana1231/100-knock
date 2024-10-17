#!/bin/bash
MAIN=path/to/main

export CUDA_VISIBLE_DEVICES=1,2

fairseq-train ${MAIN}/data/subword/ \
    --task translation \
    --arch transformer \
    --source-lang ja --target-lang en \
    --max-epoch 10 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --batch-size 64 \
    --update-freq 2 \
    --dropout 0.2 --weight-decay 1e-4 \
    --optimizer adam --clip-norm 1.0 --adam-betas '(0.9, 0.98)' \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 200000 --save-dir model96/ \
    --fp16 --wandb-project 100knock_96 > ${MAIN}/logs/train96.log
