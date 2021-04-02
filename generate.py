#!/usr/bin/env python
# coding: utf-8

# In[57]:


import os
import re
import numpy as np
import nltk
import scipy
import json

from nltk import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from rich.progress import track
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# Parameters
min_word_freq = 5
win_size = 3
embed_dim = 100

data_path = "/home/data-master/evancw/flatten_gigaword/tmp/" # data file in Orion
# data_path = "/home/jim/coding/Research/WordEmbed/dataset/"     # data file in WSL

from utils import *

from arnoldi import arnoldi_iteration


# In[58]:


corpus = []
with open("./tmp/line_corpus.txt", 'r') as lc:
    for line in track(lc.readlines()):
        corpus.append(line.strip().split())


# In[50]:


# Construct vocabulary & frequency statics

def get_vocab(corpus: list):
    vocab_freq = defaultdict(lambda: 0)
    for sentence in track(corpus, description="Collecting words..."):
        for word in sentence:
            vocab_freq[word] += 1
    return vocab_freq


def get_freq(w: str):
    return vocab_freq[w]

if not os.path.exists('./tmp/nyt_vocab.json'):
    vocab_freq = get_vocab(corpus)
    vocab = [word for word in vocab_freq.keys() if
             get_freq(word) >= min_word_freq]
    vocab = {word:i for i, word in enumerate(vocab)}
    with open('./tmp/nyt_vocab.json', 'w') as f:
        json.dump(vocab, f, indent=6)
    with open('./tmp/nyt_vocab_freq.json', 'w') as f:
        json.dump(vocab_freq, f, indent=6)
else:
    with open('./tmp/nyt_vocab.json', 'r') as f:
        vocab = json.load(f)
    with open('./tmp/nyt_vocab_freq.json', 'r') as f:
        vocab_freq = json.load(f)

vocab_len = len(vocab)

print("# vocab:\t", vocab_len)


# In[51]:


# constrain the length of vocab to be 300,000.
max_vocab = 300000
sorted_vocab = dict(sorted(vocab_freq.items(), key=lambda x: x[1], reverse=True))
freq_vocab_set = set(list(sorted_vocab)[:max_vocab])
freq_vocab = {x: vocab[x] for x in freq_vocab_set}
freq_vocab = dict(sorted(freq_vocab.items(), key=lambda x:x[1]))

# rearrange the new vocab (freq_vocab)
vocab_ = {k: i for i, k in enumerate(freq_vocab.keys())}
len(vocab_)


# In[52]:


# Construct co-occurence matrix
# Note: The comatrix contains all the vocab in the corpus. i.e. It does not neglect any Out of Vocabulary word. To constrain the vocabulary,
#       use the subset of the matrix M
from collections import defaultdict


def get_comatrix(corpus, win_size, word_dict):
    vocab_len = len(word_dict)
    coo = defaultdict(lambda: 0)
    for sent in track(corpus, description="Extracting Co-Occurrence Matrix...\t"):
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
    W = scipy.sparse.csr_matrix((data, coordinate), shape=(vocab_len, vocab_len), dtype=np.float32)
    return W

# construct&save/load Co-occurence Matrix
import scipy
if not os.path.exists(f"./tmp/nyt_M_{win_size}.npz"):
    M = get_comatrix(corpus, win_size, vocab)
    scipy.sparse.save_npz(f'./tmp/nyt_M_{win_size}.npz', M)
else:
    M = scipy.sparse.load_npz(f"./tmp/nyt_M_{win_size}.npz")


# In[53]:


# Capture the comatrix w.r.t. the most frequent <max_vocab> words.
indices = np.array(list(freq_vocab.values()))
M_ = (M[indices].T)[indices].T
M_.shape


# In[54]:


# Square root of the matrix M before Arnoldi iteration
sqrt_M = M_.sqrt()


# In[55]:


# Construct word embeddings by Arnoldi iteration
if os.path.exists(f"./tmp/nyt_Q_{win_size}.npy"):
    Q = np.load(f"./tmp/nyt_Q_{win_size}.npy")
else:
    b = np.random.random(size=max_vocab)  # initial vector
    Q, h = arnoldi_iteration(sqrt_M, b, embed_dim)
    np.save(f"./tmp/nyt_Q_{win_size}.npy", Q)  # save Word embeddings


# In[ ]:


from gensim.models.keyedvectors import KeyedVectors


# In[56]:


dim = embed_dim
Q_ = Q[:, :dim]
WE_ = normalize(Q_, axis=1, norm="l2")
kv = KeyedVectors(vector_size=dim)
kv.add(list(vocab_.keys()), WE_)
kv.save_word2vec_format(f"./tmp/arnodi_{dim}_{win_size}.kv")

