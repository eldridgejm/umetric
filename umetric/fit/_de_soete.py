import numpy as np
import scipy.optimize
import collections

DeSoeteResult = collections.namedtuple("DeSoeteResult", 
                                        ["ultrametric", "loss", "penalty", 
                                         "n_iters"])

def closest_l_2_de_soete(metric, maxiter=100, convergence=1e-10, d_init=None,
                         method="Powell"):
    """Computes the closest ultrametric in l_2 by sequentially minimizing an
    unconstrained objective function."""
    n = metric.shape[0]

    # the objective function
    phi = lambda x, gamma: loss(x, metric) + gamma*penalty(x)

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
        gamma = loss(metric, d_init) / penalty(d_init)

    while True:
        # design the objective function to take a vector instead of a matrix
        obj = lambda x: phi(np.reshape(x, (n,n)), gamma)

        # now optimize
        res = scipy.optimize.minimize(obj, d_init.flatten(), method=method)
        d_opt = np.reshape(res.x, (n,n))

        delta = (d_opt - d_init)**2
        delta = delta[np.triu_indices_from(delta)].sum()

        if delta < convergence or q >= maxiter:
            ultrametric = np.triu(d_opt, k=1)
            ultrametric = ultrametric + ultrametric.T

            return DeSoeteResult(ultrametric, 
                                 loss(ultrametric, metric),
                                 penalty(ultrametric),
                                 q)
        else:
            d_init = d_opt
            q += 1
            gamma *= 10


def loss(x,y):
    """Computes the squared error between two metrics described as square
    matrices. Only the upper triangle of each metric is looked at."""
    inds = np.triu_indices_from(x)
    return np.sum((x[inds] - y[inds])**2)


def penalty(d):
    """Computes the penalty which enforces ultrametricity in the 
    optimization."""
    # omega is the set of (i,j,k) for which d_{ij} <= min(d_)
    i,j,k = non_ultrametric_triples(d).T
    d_ik = d[i,k]
    d_jk = d[j,k]
    
    return np.sum((d_ik - d_jk)**2)


def non_ultrametric_triples(d):
    """Returns all 3-tuples (i,j,k) for which d[i,j] < min(d[i,k], d[j,k]).
    Assumes a symmetric input matrix."""
    n = d.shape[0]
    grid = np.mgrid[:n, :n, :n]
    inds = np.column_stack(x.flatten() for x in grid)
    
    inds = inds[inds[:,0] < inds[:,1]]
    inds = inds[inds[:,0] < inds[:,2]]
    inds = inds[inds[:,1] < inds[:,2]]
    
    i,j,k = inds.T
    
    d_ij = d[i,j]
    d_ik = d[i,k]
    d_jk = d[j,k]
    
    non_um_inds = d_ij < np.min(np.column_stack((d_ik, d_jk)), axis=1)
    non_um_inds = np.logical_and(non_um_inds, d_ik != d_jk)
    return inds[non_um_inds]


def shake_dissimilarity(d):
    """Randomly shakes the dissimilarity."""
    n = d.shape[0]
    delta = (d - d.mean())**2
    variance = (2./(3*n*(n-1))*delta[np.triu_indices_from(delta)].sum())
    
    eps = np.random.normal(0, np.sqrt(variance), (n,n))
    return d + eps
