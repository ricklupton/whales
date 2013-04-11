import utils
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

def test_skew():
    assert np.allclose(utils.skew(np.zeros(3)), np.zeros((3,3)))
    assert np.allclose(utils.skew([1,2,3]), np.array([
        [ 0, -3,  2],
        [ 3,  0, -1],
        [-2,  1,  0],
    ]))

def test_shift():
    A = np.arange(1, 5)
    assert np.allclose(utils.shift(A,  0), A)
    assert np.allclose(utils.shift(A,  2, fill=0), [3, 4, 0, 0])
    assert np.allclose(utils.shift(A, -2, fill=0), [0, 0, 1, 2])
    assert np.allclose(utils.shift(A,  2),         [3, 4, 4, 4])
    assert np.allclose(utils.shift(A, -2),         [1, 1, 1, 2])

def test_output_variance():
    w = np.arange(0, 5, 0.5)

    # Zero variance with zero transfer function
    H = np.zeros((10, 3, 3)) # frequency, another parameter, xyz
    S = np.ones(10)
    assert_array_equal(utils.output_variance(w, H, S), np.zeros((3,3,3)))

    # Zero variance with zero spectrum
    H = np.ones((10, 3, 3)) # frequency, another parameter, xyz
    S = np.zeros(10)
    assert_array_equal(utils.output_variance(w, H, S), np.zeros((3,3,3)))

    # White noise spectrum, single peak in transfer function
    H = np.zeros((10, 3, 3))
    S = np.ones(10)
    # Put the peaks at various frequencies, amplitudes and coordinates
    H[1,0,:] = 1.1
    H[4,1,2] = 2.3
    H[6,2,0] = 3.4
    s1 = utils.output_variance(w, H, S)

    # Expected:
    dw = 0.5
    s = np.zeros((3,3,3))
    s[0,:,:] = 1.1**2 * dw
    s[1,2,2] = 2.3**2 * dw
    s[2,0,0] = 3.4**2 * dw
    assert_allclose(s1[0], s[0])
    assert_allclose(s1[1], s[1])
    assert_allclose(s1[2], s[2])

def test_output_variance_complex():
    assert False
    # Check conjugate is in right place
