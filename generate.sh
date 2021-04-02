#!/usr/bin/env bash
source env/bin/activate
for dim in 300 500
    do
        for win_size in 2 3 5
            do
                echo "Start generating dim=$dim win_size=$win_size embeddings."
                nohup python generate.py --win_size $win_size --dim $dim --tag nyt &
            done
    done