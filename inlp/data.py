"""
Extract three groups of words from the word embeddings according to the gender components.
    - female
    - male
    - neutral
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def extract_train_set(W: np.ndarray, vocab: dict, num_per_group: int = 750):
    pairs = [("male", "female"), ("masculine", "feminine"),
             ("he", "she"), ("him", "her")]
    gender_vecs = [W[vocab[p[0]]] - W[vocab[p[1]]] for p in pairs]
    pca = PCA(n_components=1)
    pca.fit(gender_vecs)
    gender_direction = pca.components_[0]
    gender_unit = gender_direction / np.linalg.norm(gender_direction)
    gender_scores = W @ gender_unit

    neutral_scores = np.abs(gender_scores)
    sorted_idx = np.argsort(gender_scores)

    male_idx = sorted_idx[-num_per_group:]
    female_idx = sorted_idx[:num_per_group]
    neutral_idx = np.argsort(neutral_scores)[:num_per_group]

    idx = np.concatenate((male_idx, female_idx, neutral_idx))
    X = W[idx]
    Y = np.concatenate((
        np.ones(num_per_group, dtype=int),
        np.zeros(num_per_group, dtype=int),
        -np.ones(num_per_group, dtype=int)
    ))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42)

    return X_train, X_test, Y_train, Y_test
