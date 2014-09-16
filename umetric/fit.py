import numpy as _np
from . import core as _core

def closest_l_infinity(metric):
    """Find the optimal ultrametric in l-infinity."""
    m_star = _core.linkage_ultrametric(metric, method="single")
    ultrametric = m_star + 1./2 * _np.max(metric - m_star)
    ultrametric[_np.diag_indices_from(ultrametric)] = 0.
    return ultrametric


def closest_l_2_heuristic(metric):
    """Uses a heuristic to approximate the closest ultrametric in l2."""
    m_star = _core.linkage_ultrametric(metric, method="single")

    # compute the average difference between the metric and the maximal 
    # subdominant ultrametric in the upper triangle of the matrix
    inds = _np.triu_indices_from(metric)
    alpha = (metric[inds] - m_star[inds]).mean()

    ultrametric = m_star + alpha
    ultrametric[_np.diag_indices_from(ultrametric)] = 0.

    return ultrametric
