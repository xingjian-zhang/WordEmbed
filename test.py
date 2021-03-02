import argparse
import json
import os
from itertools import product
from utils import word_similarity

import numpy as np
from rich.progress import track
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser()
parser.add_argument("--corpora", type=str, default="brown")
parser.add_argument("--embed_dim", type=int, default=50)
parser.add_argument("--window_size", type=int, default=5)
parser.add_argument("--min_word_freq", type=int, default=10)
parser.add_argument("--test_config", type=str, default=None)
args = parser.parse_args()


def summary(embed_dir,
            center_word: list = [],
            n_closest: int = 0):
    # check if the embeddings exist
    if not os.path.exists(embed_dir):
        return

    # load the embeddings and config
    config_file = os.path.join(embed_dir, "config.json")
    vocab_file = os.path.join(embed_dir, "vocab.json")
    embed_file = os.path.join(embed_dir, "Q.npy")
    with open(config_file, 'r') as config_:
        config = json.load(config_)
    with open(vocab_file, 'r') as vocab_:
        vocab = json.load(vocab_)
    Q = np.load(embed_file)

    # print the summary of the embeddings Q
    print("-" * 25)
    print("{:<15} {:<10}".format("Attribute", "Value"))
    print("-" * 25)
    for k, v in config.items():
        print("{:<15} {:<10}".format(k, v))
    print("{:<15} {:<10}".format("vocab_len", len(vocab)))
    print("-" * 25)

    inv_vocab = {v: k for k, v in vocab.items()}
    Q_ = normalize(Q, axis=1, norm='l2')
    for word in center_word:
        rank = word_similarity(Q_, vocab, inv_vocab, word)
        closest_words = list(rank.keys())
        print(word, closest_words)

    # idx = np.random.randint(0, len(vocab))
    # print(inv_vocab[idx], Q_[idx])


def grid_test(corpora: list,
              embed_dim: list,
              window_size: list,
              min_word_freq: list,
              center_word: list,
              n_closest: int):
    test_queue = list(product(corpora, embed_dim, window_size, min_word_freq))
    for c, e, w, m in track(test_queue, description="Testing..."):
        embed_dir = f"embed/{c}/{e}/{w}/{m}"
        summary(embed_dir, center_word, n_closest)


def main():
    if args.test_config is None:
        embed_dir = f"embed/{args.corpora}/{args.embed_dim}/{args.window_size}/{args.min_word_freq}"
        summary(embed_dir)
    else:
        with open(args.test_config, 'r') as tc:
            test_config = json.load(tc)
            grid_test(**test_config)


main()
