import numpy as np
from scipy import integrate

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

def output_variance(w, H, S):
    """Calculate the variance of the output response described by the TF ``H``

    ``w`` is the frequency in rad/s
    ``H`` is the transfer function matrix, frequency along first axis
    ``S`` is the spectrum, shape (w,)

    $$ \sigma_{ij}^2 = \mathrm{Re} \int_0^{\infty} H_i(\omega)
    H_j(\omega) S_{\eta\eta}(\omega) \;\mathrm{d}\omega \qquad (i, j = 1, 2)$$
    """
    assert len(H.shape) >= 2
    assert H.shape[0] == len(w) == len(S)
    y = np.einsum('w...i,w...j,w...->w...ij', H, H.conj(), S)
    s = integrate.trapz(y, x=w, axis=0).real
    return s
