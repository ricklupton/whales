from collections import namedtuple
import numpy as np
from numpy import array, newaxis
import scipy.linalg

BaseLinearSystem = namedtuple('LinearSystem', ['M', 'C', 'K'])
class LinearSystem(BaseLinearSystem):
    def __add__(self, other):
        return LinearSystem(*[a+b for a, b in zip(self, other)])


def transfer_function(w, system):
    """Calculate transfer functions from LinearSystem"""
    def raise_shape(x):
        assert x.ndim in (2, 3)
        return x if x.ndim == 3 else x[newaxis]
    M, C, K = map(raise_shape, system)
    w3d = w[:, newaxis, newaxis]
    F = -(w3d**2)*M + 1j*w3d*C + K
    H = array([scipy.linalg.inv(F[i]) for i in range(len(w))])
    return H


def RAOs(w, H, X):
    assert H.shape[0] == X.shape[0] == len(w)
    assert H.shape[2] == X.shape[1]
    H_wave = np.einsum('wij, wj -> wi', H, X)
    return H_wave


#__all__ = (LinearSystem, 'rigid_bodies', 'cylinder', 'transfer_function')
