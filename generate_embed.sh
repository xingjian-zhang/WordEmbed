#!/bin/zsh

set -Eeuo pipefail

for ws in 2 3 4 5 6; do
    for dim in 10 30 50 100 200; do
        for min in 3 5 7 10; do
            echo -e "window size $ws  dim $dim  min word freq $min"
            python train.py --window_size $ws --embed_dim $dim --min_word_freq $min --num_samples 0
            echo -e "\n"
        done
    done
done