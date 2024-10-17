#!/bin/bash
MAIN=path/to/main

export CUDA_VISIBLE_DEVICES=0,1

fairseq-train ${MAIN}/data/subword/ \
    --task translation \
    --arch transformer \
    --source-lang ja --target-lang en \
    --max-epoch 30 \
    --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --batch-size 64 \
    --update-freq 2 \
    --dropout 0.2 --weight-decay 1e-4 \
    --optimizer adam --clip-norm 1.0 --adam-betas '(0.9, 0.98)' \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 300000 --save-dir model97/ --patience 4 \
    --fp16 --wandb-project 100knock_96 > ${MAIN}/logs/train.log

fairseq-interactive --path ${MAIN}/model97/checkpoint_best.pt ${MAIN}/data/subword < ${MAIN}/data/subword/test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > ${MAIN}/data/subword/eval97.out
