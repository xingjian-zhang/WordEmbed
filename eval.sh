source /home/jimmyzxj/word-embeddings-benchmarks/env/bin/activate
for dim in 300 500
    do
        for win_size in 2 3 5
            do
                echo "Start evaluating dim=$dim win_size=$win_size arnoldi embeddings."
                nohup python3 /home/jimmyzxj/word-embeddings-benchmarks/scripts/evaluate_on_all.py -f /home/jimmyzxj/WordEmbed/tmp/arnodi_${dim}_${win_size}.kv -o /home/jimmyzxj/word-embeddings-benchmarks/results/result_arnoldi_${dim}_${win_size}.csv -p word2vec > nohup_${dim}_${win_size}.log 2>&1 &
            done
    done