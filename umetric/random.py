import numpy as _np

def lapointe_random_uniform_ultrametric(n, prng=None):
    """Generate a uniform random ultrametric using the method of Lapointe."""
    if prng is None:
        prng = _np.random.RandomState()

    fusion_levels = prng.uniform(0,1,n-1)
    
    ultrametric = _np.zeros((n,n))
    
    current_diag_inds = _off_diagonal_indices(n,1)
    ultrametric[current_diag_inds] = fusion_levels

    for j in range(2,n):
        prev_diag_inds = current_diag_inds
        current_diag_inds = _off_diagonal_indices(n, j)

        prev_diag = ultrametric[prev_diag_inds]
        current_diag = _np.maximum(prev_diag[:-1], prev_diag[1:])

        ultrametric[current_diag_inds] = current_diag

    ultrametric = ultrametric + ultrametric.T

    i,j = _np.triu_indices_from(ultrametric)

    shuffle = prng.permutation(_np.arange(n))

    shuffled_ultrametric = _np.zeros_like(ultrametric)
    shuffled_ultrametric[i,j] = ultrametric[shuffle[i], shuffle[j]]

    return shuffled_ultrametric + shuffled_ultrametric.T


def _off_diagonal_indices(n, k):
    """Returns the indices of the k-th diagonal above the main diagonal."""
    i = _np.arange(n-k)
    j = i + k
    
    return i,j


