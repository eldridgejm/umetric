import numpy as _np
from .. import core as _core

from _de_soete import closest_l_2_de_soete

def closest_l_infinity(metric):
    """Find the optimal ultrametric in l-infinity."""
    m_star = _core.linkage_ultrametric(metric, method="single")
    return m_star + 1./2 * _np.max(metric - m_star)


def closest_l_2_heuristic(metric):
    """Uses a heuristic to approximate the closest ultrametric in l2."""
    m_star = _core.linkage_ultrametric(metric, method="single")

    # compute the average difference between the metric and the maximal 
    # subdominant ultrametric in the upper triangle of the matrix
    alpha = (metric - m_star).mean()
    return m_star + alpha
