#!/bin/bash
MAIN=path/to/main

export CUDA_VISIBLE_DEVICES=0,1

# cd ${MAIN}/data/jparacrawl
# curl -k -O https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/en/3.0/bitext/en-ja.tar.gz
# tar -xzvf en-ja.tar.gz
# cd ${MAIN}

# python ${MAIN}/get_jpara.py ${MAIN}/data/jparacrawl/en-ja/en-ja.bicleaner05.txt ${MAIN}/data/jparacrawl/jpara.ja ${MAIN}/data/jparacrawl/jpara.en

# subword-nmt apply-bpe -c ${MAIN}/data/subword/codec.ja.txt < ${MAIN}/data/jparacrawl/jpara.ja > ${MAIN}/data/jparacrawl/train.jpara.ja
# subword-nmt apply-bpe -c ${MAIN}/data/subword/codec.en.txt < ${MAIN}/data/jparacrawl/jpara.en > ${MAIN}/data/jparacrawl/train.jpara.en

# # preprocess
# fairseq-preprocess -s ja -t en \
#     --trainpref  ${MAIN}/data/jparacrawl/train.jpara \
#     --validpref ${MAIN}/data/subword/dev.sub \
#     --testpref ${MAIN}/data/subword/test.sub \
#     --destdir ${MAIN}/data/jparacrawl/ \
#     --task translation

# 91
fairseq-train ${MAIN}/data/jparacrawl/ \
    --task translation \
    --arch transformer \
    --source-lang ja --target-lang en \
    --max-epoch 20 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --batch-size 32 \
    --update-freq 2 \
    --dropout 0.2 --weight-decay 1e-4 \
    --optimizer adam --clip-norm 1.0 --adam-betas '(0.9, 0.98)' \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 10000 --save-dir model98/ --patience 4 \
    --fp16 --wandb-project 100knock > ${MAIN}/logs/train.log

# 92
fairseq-interactive --path ${MAIN}/model98/checkpoint_best.pt ${MAIN}/data/jparacrawl < ${MAIN}/data/subword/test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > ${MAIN}/data/jparacrawl/eval98.out

# 93
fairseq-score \
    --sys ${MAIN}/data/jparacrawl/eval98.out \
    --ref ${MAIN}/data/subword/test.en

# 94
for N in {1..5} ; do
    fairseq-interactive --path ${MAIN}/model98/checkpoint_best.pt --beam ${N} ${MAIN}/data/jparacrawl < ${MAIN}/data/subword/test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > ${MAIN}/data/jparacrawl/eval98.${N}.out
done

for N in {1..5} ; do
    fairseq-score --sys ${MAIN}/data/jparacrawl/eval98.${N}.out --ref ${MAIN}/data/subword/test.sub.en > ${MAIN}/data/jparacrawl/eval98.${N}.score
done

python ${MAIN}/score.py ${MAIN}/data/jparacrawl plt98.png



