#!/bin/sh
CUDA_VISIBLE_DEVICES=3 python -u run.py --do_train --cuda \
    --do_valid \
    --do_test \
    --index_path processed/bert_msra_index.pkl \
    --model bert \
    --train_data_path processed/bert_msra_train_data.pkl \
    --valid_data_path processed/bert_msra_test_data.pkl \
    --trainable_embedding \
    -lr 0.00005 -e 20 \
    -save models/bert 