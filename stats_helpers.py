from __future__ import division
import numpy as np
from numpy import pi, inf
from scipy.integrate import quad, dblquad

def normal_distribution(x, sigma):
    """Return the PDF of a normal distribution evaluated at x"""
    return 1/(sigma*np.sqrt(2*pi)) * np.exp(-x**2 / (2*sigma**2))

def binormal_distribution(x, y, sigma):
    """Return the PDF of a binormal distribution evaluated at x & y.
    sigma is a 2x2 matrix of standard deviations.
    """
    r = (sigma[0,1]**2 / (sigma[0,0]*sigma[1,1]))
    D = 2*pi*sigma[0,0]*sigma[1,1]*(1-r**2)**0.5
    return (1/D) * np.exp(-(y**2 * sigma[0,0]**2 -
                            2*r*sigma[0,0]*sigma[1,1]*x*y +
                            x**2 * sigma[1,1]**2) /
                          (2*sigma[0,0]**2 * sigma[1,1]**2 * (1-r**2)))

def normal_expectation(f, sigma):
    """Calculate E[f(x)] for the normal distribution with sigma"""
    # Calculate for each element of vector-valued function
    result = np.asarray(f(0))
    integrand = lambda x: np.array(f(x)) * normal_distribution(x, sigma)
    if result.ndim == 0:
        return quad(integrand, -inf, inf)[0]
    else:
        return np.array([quad(lambda x: integrand(x).flat[i], -inf, inf)[0]
                         for i in range(result.size)]).reshape(result.shape)

def binorm_expectation(f, sigma):
    """Calculate E[f(x,y)] for the binormal distribution with sigma_ij"""
    if sigma[0,0] == 0:
        return normal_expectation(lambda y: f(np.array([0, y])), sigma[1,1])
    elif sigma[1,1] == 0:
        return normal_expectation(lambda x: f(np.array([x, 0])), sigma[0,0])

    def integrand(x, y):
        return np.array(f(np.array([x,y]))) * binormal_distribution(x, y, sigma)
    def dblquad_inf(g):
        return dblquad(g, -inf, inf, lambda x: -inf, lambda x: inf)[0]

    # Calculate for each element of vector-valued function
    result = np.asarray(f(np.zeros(2)))
    if result.ndim == 0:
        return dblquad_inf(integrand)
    else:
        return np.array([dblquad_inf(lambda x,y: integrand(x,y).flat[i])
                         for i in range(result.size)]).reshape(result.shape)

def tests():
    assert np.allclose(binorm_expectation(lambda xy: [1,1], np.array([[1,0],[0,0]])), [1,1])
    assert np.allclose(binorm_expectation(lambda xy: 1,     np.array([[1,0],[0,0]])), 1    )
    assert np.allclose(binorm_expectation(lambda xy: [1,1], np.array([[1,0],[0,1]])), [1,1])
    assert np.allclose(binorm_expectation(lambda xy: xy,    np.array([[1,0],[0,0]])), [0,0])
