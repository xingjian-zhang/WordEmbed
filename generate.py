# coding: utf-8


import json
import os
import argparse
from collections import defaultdict
import logging
import time
from re import DEBUG

import numpy as np
import scipy
from gensim.models.keyedvectors import KeyedVectors
from rich.progress import track
from sklearn.preprocessing import normalize

from arnoldi import arnoldi_iteration
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--min_freq", type=int, default=5)
parser.add_argument("--win_size", type=int, default=3)
parser.add_argument("--dim", type=int, default=100)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--max_vocab", type=int, default=300000)
parser.add_argument("--corpus", type=str, default="./tmp/line_corpus.txt",
                    help="Path to corpus (a single file).")
parser.add_argument("--tag", type=str, default="nyt",
                    help="Name/Tag of the embeddings. It is used to name the generated files.")
args = parser.parse_args()

# Parameters
min_word_freq = args.min_freq
win_size = args.win_size
embed_dim = args.dim
max_vocab = args.max_vocab
corpus_path = args.corpus
tag = args.tag

# Config logger
if args.output_dir is not None:
    output_dir = args.output_dir
else:
    output_dir = "log/arnoldi/"
os.makedirs(output_dir, exist_ok=True)
filename = "arnoldi_{}_{}_{}.log".format(
    win_size, embed_dim, int(time.time())
)
filename = os.path.join(output_dir, filename)
logging.basicConfig(
    level=DEBUG, format="%(asctime)s %(message)s", filename=filename)

logging.info("Start reading from corpus %s.", corpus_path)
corpus = []
try:
    with open(corpus_path, 'r') as lc:
        for line in lc.readlines():
            corpus.append(line.strip().split())
except FileNotFoundError as e:
    logging.error("Failed to open the corpus:", str(e))
    exit(0)
logging.info("Collected %d sentences from corpus.", len(corpus))
# Construct vocabulary & frequency statics


def get_vocab(corpus: list):
    vocab_freq = defaultdict(lambda: 0)
    for sentence in corpus:
        for word in sentence:
            vocab_freq[word] += 1
    return vocab_freq


def get_freq(w: str):
    return vocab_freq[w]


vocab_file = f"./tmp/{tag}_vocab.json"
vocab_freq_file = f"./tmp/{tag}_vocab_freq.json"
if not os.path.exists(vocab_file):
    logging.info("I did not find the vocab file. Yet I will create a new one.")
    vocab_freq = get_vocab(corpus)
    vocab = [word for word in vocab_freq.keys() if
             get_freq(word) >= min_word_freq]
    vocab = {word: i for i, word in enumerate(vocab)}
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f, indent=6)
    logging.info("Successfully dumped vocab file to %s.", vocab_file)
    with open(vocab_freq_file, 'w') as f:
        json.dump(vocab_freq, f, indent=6)
    logging.info("Successfully dumped vocab freq file to %s.", vocab_file)
else:
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
        logging.info("Successfully loaded vocab from %s.", vocab_file)
    with open(vocab_freq_file, 'r') as f:
        vocab_freq = json.load(f)
        logging.info("Successfully loaded vocab freq from %s.", vocab_freq_file)

vocab_len = len(vocab)
logging.info("Extract %d words from the corpus.", vocab_len)

# constrain the length of vocab to be 300,000.
sorted_vocab = dict(
    sorted(vocab_freq.items(), key=lambda x: x[1], reverse=True))
freq_vocab_set = set(list(sorted_vocab)[:max_vocab])
freq_vocab = {x: vocab[x] for x in freq_vocab_set}
freq_vocab = dict(sorted(freq_vocab.items(), key=lambda x: x[1]))
# rearrange the new vocab (freq_vocab)
vocab_ = {k: i for i, k in enumerate(freq_vocab.keys())}
len(vocab_)


# Construct co-occurence matrix
# Note: The comatrix contains all the vocab in the corpus. i.e. It does not neglect any Out of Vocabulary word. To constrain the vocabulary,
#       use the subset of the matrix M


def get_comatrix(corpus, win_size, word_dict):
    vocab_len = len(word_dict)
    coo = defaultdict(lambda: 0)
    for sent in corpus:
        words = sent
        sent_len = len(words)
        for i in range(sent_len):
            word = words[i]
            try:
                word_idx = word_dict[word]
            except KeyError:
                continue
            win_left = max(0, i - win_size)
            win_right = min(sent_len, i + win_size)
            contexts = words[win_left:i] + words[i + 1:win_right]
            for context in contexts:
                try:
                    context_idx = word_dict[context]
                except KeyError:
                    continue
                coo[(word_idx, context_idx)] += 1
    coordinate = np.array(list(coo.keys())).T
    data = np.array(list(coo.values()))
    W = scipy.sparse.csr_matrix((data, coordinate), shape=(
        vocab_len, vocab_len), dtype=np.float32)
    return W


# construct&save/load Co-occurence Matrix
M_file = f"./tmp/{tag}_M_{win_size}.npz"
if not os.path.exists(M_file):
    logging.info("Start extracting co-occurrence matrix.")
    M = get_comatrix(corpus, win_size, vocab)
    logging.info("Successfully extracted co-occurrence matrix.")
    scipy.sparse.save_npz(M_file, M)
    logging.info("Successfully saved co-occurrence matrix to %s.", M_file)
else:
    M = scipy.sparse.load_npz(M_file)
    logging.info("Successfully loaded co-occurrence matrix from %s.", M_file)

# Capture the comatrix w.r.t. the most frequent <max_vocab> words.
indices = np.array(list(freq_vocab.values()))
M_ = (M[indices].T)[indices].T
M_.shape


# Square root of the matrix M before Arnoldi iteration
sqrt_M = M_.sqrt()


# Construct word embeddings by Arnoldi iteration
Q_file = f"./tmp/{tag}_Q_{win_size}.npy"
if os.path.exists(Q_file):
    Q = np.load(Q_file)
    logging.info("Successfully loaded embeddings matrix from %s.", Q_file)
else:
    logging.info("Start arnoldi iterations.")
    b = np.random.random(size=max_vocab)  # initial vector
    Q, h = arnoldi_iteration(sqrt_M, b, embed_dim)
    logging.info("Successfully extracted word embeddings from arnoldi iteration.")
    np.save(Q_file, Q)  # save Word embeddings
    logging.info("Successfully saved word embedding matrix Q to %s.", Q_file)


dim = embed_dim
word2vec_file = f"./tmp/arnodi_{dim}_{win_size}.kv"
Q_ = Q[:, :dim]
we_ = normalize(Q_, axis=1, norm="l2")
kv = KeyedVectors(vector_size=dim)
kv.add(list(vocab_.keys()), we_)
kv.save_word2vec_format(word2vec_file)
logging.info("Successfully saved word embeddings of dimension %d to %s.", dim, word2vec_file)
