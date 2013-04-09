import utils
import numpy as np

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
