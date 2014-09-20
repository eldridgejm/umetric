import os as _os
import numpy as _np


def load_kaufman_flowers():
    filepath = _os.path.join(_os.path.dirname(__file__), "kaufman.npy")
    return _np.load(filepath)


def load_ruspini():
    filepath = _os.path.join(_os.path.dirname(__file__), "ruspini.npy")
    return _np.load(filepath)


def load_shepard_digits():
    filepath = _os.path.join(_os.path.dirname(__file__), "shepard.npy")
    return _np.load(filepath)


def toy_metric():
    return _np.array([4,2,1,2,1,1])
