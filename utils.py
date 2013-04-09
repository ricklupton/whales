import numpy as np

def skew(x):
    """Return the skew matrix of the vector x"""
    return np.array([
        [ 0,    -x[2],  x[1]],
        [ x[2],  0,    -x[0]],
        [-x[1],  x[0],  0   ],
    ])

def shift(A, n, fill=None):
    """Return A shifted n places left"""
    if fill is None:
        fill = A[-1] if n >= 0 else A[0]
    if n >= 0:
        return np.r_[ A[n:], fill * np.ones(n) ]
    else:
        return np.r_[ fill * np.ones(-n), A[:n] ]
