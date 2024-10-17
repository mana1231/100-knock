#!/bin/bash

# wget https://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
# tar zxvf kftt-data-1.0.tar.gz
# git clone https://github.com/pytorch/fairseq
# pip install --editable fairseq/

MAIN=path/to/main

for name in train dev test; do
  python ${MAIN}/preprocess.py ${MAIN}/data/orig/kyoto-${name}.ja ${MAIN}/data/preprocess/${name}.ja ja
  python ${MAIN}/preprocess.py ${MAIN}/data/orig/kyoto-${name}.en ${MAIN}/data/preprocess/${name}.en en
done

fairseq-preprocess -s ja -t en \
    --trainpref  ${MAIN}/data/preprocess/train \
    --validpref ${MAIN}/data/preprocess/dev \
    --testpref ${MAIN}/data/preprocess/test \
    --destdir ${MAIN}/data/preprocess/ \
    --task translation