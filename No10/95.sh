#!/bin/bash
MAIN=path/to/main

export CUDA_VISIBLE_DEVICES=0

subword-nmt learn-bpe -s 10000 < ${MAIN}/data/preprocess/train.ja > ${MAIN}/data/subword/codec.ja.txt
subword-nmt learn-bpe -s 10000 < ${MAIN}/data/preprocess/train.en > ${MAIN}/data/subword/codec.en.txt

for name in train dev test; do
  subword-nmt apply-bpe -c ${MAIN}/data/subword/codec.ja.txt < ${MAIN}/data/preprocess/${name}.ja > ${MAIN}/data/subword/${name}.sub.ja
  subword-nmt apply-bpe -c ${MAIN}/data/subword/codec.en.txt < ${MAIN}/data/preprocess/${name}.en > ${MAIN}/data/subword/${name}.sub.en
done

# preprocess
fairseq-preprocess -s ja -t en \
    --trainpref  ${MAIN}/data/subword/train.sub \
    --validpref ${MAIN}/data/subword/dev.sub \
    --testpref ${MAIN}/data/subword/test.sub \
    --destdir ${MAIN}/data/subword/ \
    --task translation

# 91
fairseq-train ${MAIN}/data/subword/ \
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
    --max-tokens 10000 --save-dir model95/ \
    --fp16 > ${MAIN}/logs/train95.log

# 92
fairseq-interactive --path ${MAIN}/model95/checkpoint_best.pt ${MAIN}/data/subword < ${MAIN}/data/subword/test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > ${MAIN}/data/subword/eval95.out

# 93
fairseq-score \
    --sys ${MAIN}/data/subword/eval95.out \
    --ref ${MAIN}/data/subword/test.en

# 94
for N in `seq 1 5` ; do
    fairseq-interactive --path ${MAIN}/model95/checkpoint_best.pt --beam $N ${MAIN}/data/subword < ${MAIN}/data/subword/test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > ${MAIN}/data/subword/eval95.${N}.out
done

for N in `seq 1 5` ; do
    fairseq-score --sys ${MAIN}/data/subword/eval95.${N}.out --ref ${MAIN}/data/subword/test.sub.en > ${MAIN}/data/subword/eval95.${N}.score
done

python ${MAIN}/score.py ${MAIN}/data/subword plt95.png



