#!/bin/bash
MAIN=path/to/main

# for N in {1..5} ; do
#     fairseq-interactive --path ${MAIN}/model91/checkpoint_best.pt --beam ${N} ${MAIN}/data/preprocess < ${MAIN}/data/preprocess/test.ja | grep '^H' | cut -f3 > ${MAIN}/data/preprocess/eval.${N}.out
# done

# for N in {1..5} ; do
#     fairseq-score --sys ${MAIN}/data/preprocess/eval.${N}.out --ref ${MAIN}/data/preprocess/test.en > ${MAIN}/data/preprocess/eval.${N}.score
# done

python ${MAIN}/score.py ${MAIN}/data/preprocess plt94.png