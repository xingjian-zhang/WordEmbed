import json
import os
from utils import get_comatrix, preprocess
from time import sleep, time
from rich.progress import track
from sklearn.feature_extraction.text import CountVectorizer
from arnoldi import arnoldi_iteration
from nltk.corpus import *
import numpy as np
import nltk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--corpora", type=str, default="brown")
# parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=0)
# parser.add_argument("--log_level", type=int, default=10)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--window_size", type=int, default=5)
parser.add_argument("--embed_dim", type=int, default=100)
parser.add_argument("--corpora_length", type=int, default=-1)
parser.add_argument("--min_word_freq", type=int, default=10)
args = parser.parse_args()

tic = time()

try:
    nltk.data.find(f"corpora/{args.corpora}")
except LookupError:
    nltk.download(args.corpora)
    sleep(3)

corpora = eval(args.corpora)
if args.corpora_length == -1:
    corpora = corpora.sents()
else:
    corpora = corpora.sents()[:args.corpora_length]
corpora = [" ".join(sent) for sent in corpora]
rs = np.random.RandomState(args.random_seed)

clean_corpora = []
corpora_len = len(corpora)
for sent in track(corpora, description=f"Preprocessing Corpora {args.corpora}...\t"):
    clean_corpora.append(
        preprocess(sent)
    )

if args.num_samples > 0:
    print("\n======================================Sample=Sentence======================================\n")
    sample_idx = np.random.choice(
        corpora_len, size=args.num_samples, replace=False)
    for i in sample_idx:
        print(i + 1, "\t ", corpora[i][:75] + "...")
        print("\t ", clean_corpora[i][:75] + "...\n")
    print("===========================================================================================\n")

del corpora
cv = CountVectorizer()
cv_fit = cv.fit_transform(clean_corpora)
vocab = cv.vocabulary_
vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
vocab_freq = cv_fit.sum(axis=0)
vocab_freq = np.array(vocab_freq).reshape(-1)
high_freq = vocab_freq >= args.min_word_freq
high_freq_words = np.array(list(vocab.keys()))[high_freq]
vocab = {k: i for i, k in enumerate(high_freq_words)}
vocab_len = len(vocab)

M = get_comatrix(clean_corpora, args.window_size, vocab)

b = rs.random(size=vocab_len)

Q, h = arnoldi_iteration(M, b, args.embed_dim)

save_dir = f"embed/{args.corpora}/{args.embed_dim}/{args.window_size}/{args.min_word_freq}"
os.makedirs(save_dir, exist_ok=True)
np.save(f"{save_dir}/Q.npy", Q)
json.dump(vocab, open(f"{save_dir}/vocab.json", 'w'))
json.dump(vars(args), open(f"{save_dir}/config.json", 'w'))
toc = time()

print(f"The vocabulary contains {vocab_len} words.", end=' ')
print(f"Finished in {toc-tic:.1f}s.")
