from scipy.special import jn, hankel1, hankel2


def hankel1d(n, z):
    """Derivative of hankel1 (from Wolfram MathWorld)"""
    return (n * hankel1(n, z) / z) - hankel1(n+1, z)


def hankel2d(n, z):
    """Derivative of hankel2 (from Wolfram MathWorld)"""
    return 0.5 * (hankel2(n-1, z) - hankel2(n+1, z))


def jnd(v, z):
    "Derivative of Bessel function jn"
    return -jn(1, z) if v == 0 else (jn(v-1, z) - jn(v+1, z)) / 2
