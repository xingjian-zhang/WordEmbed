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


def get_new_arnoldi_vec(Q: np.ndarray, M: np.ndarray):
    v, k = Q.shape
    if k == 0:
        q = np.random.rand(v)
        return q / np.linalg.norm(q)

    q = M @ Q[-1]
    rank = np.linalg.matrix_rank(Q)
    Q_ortho, _ = np.linalg.qr(Q)
    Q_ortho = Q_ortho.T[:rank]

    for q_ in Q_ortho:
        q -= q @ q_ * q_

    return q / np.linalg.norm(q)
