#!/bin/sh
CUDA_VISIBLE_DEVICES=3 python -u run.py --do_train --cuda \
        --do_valid \
    --do_test \
    --index_path processed/nn_msra_index.pkl \
        --model rnn \
    --train_data_path processed/nn_msra_train_data.pkl \
    --valid_data_path processed/nn_msra_test_data.pkl \
    --trainable_embedding \
    -lr 0.001 -e 20 \
    -save models/lstm \
           --use_crf