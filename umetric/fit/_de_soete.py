import numpy as np
import scipy.optimize
import collections

from .. import core as _core

DeSoeteResult = collections.namedtuple("DeSoeteResult", 
                                        ["ultrametric", "loss", "penalty", 
                                         "n_iters"])
def penalty_gradient_matrix(n, ik, jk):
    d = np.zeros((n,n))
    
    ix, counts = np.unique(ik, return_counts=True)
    d[ix,ix] = 2*counts
    
    ix, counts = np.unique(jk, return_counts=True)
    d[ix,ix] = 2*counts
    
    d[ik,jk] -= 2
    d[jk,ik] -= 2
    
    return d


def penalty(m):
    ij, ik, jk = _core.non_ultrametric_triples(m)
    return np.sum((m[ik] - m[jk])**2)


def fast_penalty(m, ik, jk):
    return np.sum((m[ik] - m[jk])**2)


def loss(x, y):
    """Compute the squared error."""
    return np.sum((x-y)**2)


def shake_dissimilarity(d):
    """Randomly shakes the dissimilarity."""
    n = d.shape[0]
    delta = (d - d.mean())**2
    variance = (2./(3*n*(n-1))*delta.sum())
    
    eps = np.random.normal(0, np.sqrt(variance), n)
    return d + eps


def closest_l_2_de_soete(metric, maxiter=100, convergence=1e-6, d_init=None, 
                         method="cg", method_options=None, mode="fast"):
    """Computes the closest ultrametric in l_2 by sequentially minimizing an
    unconstrained objective function."""
    n = metric.shape[0]
    
    if method_options is None:
        method_options = {}

    # iteration counter
    q = 1

    # the initial approximation to the metric
    if d_init is None:
        d_init = shake_dissimilarity(metric)
        
    # the initial tradeoff between loss and penalty
    initial_penalty = penalty(d_init)

    if np.isclose(0, initial_penalty):
        gamma = 1
    else:
        gamma = loss(metric, d_init) / initial_penalty

    while True:
        # design the objective function to take a vector instead of a matrix
        if mode == "fast":
            ij, ik, jk = _core.non_ultrametric_triples(metric)
            obj = lambda x: loss(x, metric) + gamma*fast_penalty(x, ik, jk)
        else:
            obj = lambda x: loss(x, metric) + gamma*penalty(x)

        res = scipy.optimize.minimize(obj, d_init, method=method, 
                                      options=method_options)
        d_opt = res.x
        
        delta = np.sum((d_opt - d_init)**2)

        if (delta < convergence) or (q >= maxiter):

            return DeSoeteResult(d_opt, 
                                 loss(d_opt, metric),
                                 penalty(d_opt),
                                 q)
        else:
            d_init = d_opt
            q += 1
            gamma *= 10
