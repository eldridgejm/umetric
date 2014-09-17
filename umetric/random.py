import numpy as _np

def lapointe_random_uniform_ultrametric(n):
    """Generate a uniform random ultrametric using the method of Lapointe."""
    fusion_levels = np.random.uniform(0,1,n-1)
    
    ultrametric = np.zeros((n,n))
    
    current_diag_inds = off_diagonal_indices(n,1)
    ultrametric[current_diag_inds] = fusion_levels

    for j in range(2,n):
        prev_diag_inds = current_diag_inds
        current_diag_inds = off_diagonal_indices(n, j)

        prev_diag = ultrametric[prev_diag_inds]
        current_diag = np.maximum(prev_diag[:-1], prev_diag[1:])

        ultrametric[current_diag_inds] = current_diag

    ultrametric = ultrametric + ultrametric.T

    i,j = np.triu_indices_from(ultrametric)

    shuffle = np.random.permutation(np.arange(n))

    shuffled_ultrametric = np.zeros_like(ultrametric)
    shuffled_ultrametric[i,j] = ultrametric[shuffle[i], shuffle[j]]

    return shuffled_ultrametric + shuffled_ultrametric.T
