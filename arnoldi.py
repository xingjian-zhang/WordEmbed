import numpy as np
from rich.progress import track
from inlp.data import extract_train_set


def arnoldi_iteration(A, b, n: int, verbose=False):
    """Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^n b}.

    Arguments
      A: m × m array
      b: initial vector (length m)
      n: dimension of Krylov subspace, must be >= 1

    Returns
      Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
      h: (n + 1) x n array, A on basis Q. It is upper Hessenberg.
    """
    m = A.shape[0]
    h = np.zeros((n + 1, n))
    Q = np.zeros((m, n + 1), dtype=np.float32)
    q = b / np.linalg.norm(b)  # Normalize the input vector
    Q[:, 0] = q  # Use it as the first Krylov vector

    for k in track(range(n),
                   description="Performing Arnoldi Iteration...\t",
                   disable=not verbose):
        v = A.dot(q)  # Generate a new candidate vector
        for j in range(k + 1):  # Subtract the projections on previous vectors
            h[j, k] = np.dot(Q[:, j].conj(), v)
            v = v - h[j, k] * Q[:, j]

        h[k + 1, k] = np.linalg.norm(v)
        eps = 1e-12  # If v is shorter than this threshold it is the zero vector
        if h[k + 1, k] > eps:  # Add the produced vector to the list, unless
            q = v / h[k + 1, k]  # the zero vector is produced.
            Q[:, k + 1] = q
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h


def get_new_arnoldi_vec(Q_: np.ndarray, M: np.ndarray, N: int = 1):
    """Get N new arnoldi vectors.

    Args:
        Q: Arnoldi matrix
        M: Source matrix
        N: Number of new vectors

    Returns:
        Q_: New matrix to be appended to Q
    """
    Q_ = Q_.copy()
    v, k = Q_.shape
    if k == 0:
        q = np.random.rand(v)
        Q_ = q.reshape(1, -1)
        if N == 1:
            return Q_
        else:
            N = N - 1
    rank = np.linalg.matrix_rank(Q_)
    Q_ortho, _ = np.linalg.qr(Q_)
    Q_ortho = Q_ortho.T[:rank]

    for _ in range(N):
        q = M @ Q_ortho[-1]  # Q[-1] or Q_ortho[-1]?
        for q_ in Q_ortho:
            q -= q @ q_ * q_
        q = q / np.linalg.norm(q)
        Q_ortho = np.stack((Q_ortho, q))

    Q_ = Q_ortho[-N:]
    return Q_


def train(M: np.ndarray, vocab: dict, N: int = 20, dim: int = 100):
    """Train netural word embeddings using Arnoldi iteration with chunked INLP

    Args:
        M: co-occurrence matrix, could be sparse
        vocab: vocabulary dictionary
        N: chunk length
        dim: dimension of the word embeddings

    Returns:
        Q: word embeddings matrix
    """
