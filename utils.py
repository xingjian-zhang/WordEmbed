import re
import numpy as np
from rich.progress import track
from nltk.tokenize import RegexpTokenizer


def preprocess(sentence):
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
    W = np.zeros((vocab_len, vocab_len), dtype=np.float32)
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
                W[word_idx, context_idx] += 1
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
