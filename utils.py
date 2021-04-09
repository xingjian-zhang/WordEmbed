import re
from collections import defaultdict

import numpy as np
import scipy.sparse as sparse
from nltk import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from rich.progress import track


def preprocess(file_path):
    with open(file_path, 'r') as text_file:
        data = text_file.read().replace('\n', '')
        sentences = []
        for sent in track(sent_tokenize(data), f"Preprocessing {file_path[-15:]}"):
            sentences.append(preprocess_sent(sent))
    return sentences


def preprocess_sent(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    return " ".join(tokens)


def get_comatrix(corpus, win_size, word_dict):
    vocab_len = len(word_dict)
    coo = defaultdict(lambda: 0)
    for sent in track(corpus, description="Extracting Co-Occurrence Matrix...\t"):
        words = sent.split()
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
    W = sparse.csr_matrix((data, coordinate), shape=(
        vocab_len, vocab_len), dtype=np.float32)
    return W


def word_similarity(Q, vocab, inv_vocab, w: str, n: int = 10):
    try:
        idx = vocab[w]
    except KeyError:
        print("Not found", w)
        return {}
    inner = Q @ Q[idx]
    n += 1
    closest = np.argpartition(inner, -n)[-n:]
    closest = closest[np.argsort(inner[closest])][:-1]
    return {inv_vocab[cidx]: inner[cidx] for cidx in np.flip(closest)}
