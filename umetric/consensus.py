import numpy as _np

def min_max_l_infinity(profile):
    """Minimizes the max l-infinity distance from the consensus to the
    ultrametrics in the profile."""
    sup = _np.max(profile, axis=2)
    inf = _np.min(profile, axis=2)
    consensus = sup - 1./2*_np.max(sup - inf)
    consensus[_np.diag_indices_from(consensus)] = 0.
    return consensus

def best_in_profile(profile, cost_function):
    """Applies the cost function to each ultrametric in the profile, and 
    returns the best. The cost function must return a nxn consensus ultrametric
    and return a scalar value representing its cost."""
    profile_iter = (row for row in _np.rollaxis(profile,2))
    return min(profile_iter, key=cost_function)

def sum_of_squared_errors(profile, consensus):
    """Computes the sum of squared errors between the profile and the 
    consensus."""
    return _np.sum((profile - consensus[:,:,_np.newaxis])**2)
