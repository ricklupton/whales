"""
Wave spectra
"""

import numpy as np
from numpy import pi

g = 9.81

#### Wave spectra
class PiersonMoskowitz(object):
    def __init__(self, Hs):
        """One-parameter Pierson-Moskowitz spectrum"""
        self.alpha = 0.0081
        self.beta = 0.74
        self.Hs = Hs
        # peak frequency from significant wave height
        self.wp = 1.433 * Hs**(-0.5)

    def __call__(self, w):
        S = self.alpha * g**2 / w**5 * np.exp(-self.beta * (self.wp / w)**4)
        return S


class ISSC(object):
    def __init__(self, Hs, Tz):
        """Two-parameter ISSC spectrum (Langley1987)"""
        self.Hs = Hs
        self.Tz = Tz

    def __call__(self, w):
        S = (173.0 * self.Hs**2 / (self.Tz**4 * w**5) *
             np.exp(-691.0 / (self.Tz * w)**4))
        return S


class JONSWAP(object):
    def __init__(self, Hs, Tp, gamma=None):
        """Three-parameter JONSWAP spectrum"""
        self.Hs = Hs
        self.Tp = Tp
        self.gamma = gamma or JONSWAP.default_peak(Hs, Tp)
        self.wp = 2*np.pi / Tp
        # From Naess & Moan, 2013, eq (8.18)
        self.alpha = 5.058 * Hs**2 / Tp**4 * (1 - 0.287 * np.log(self.gamma))

    def __call__(self, w):
        sigma = np.choose(w > self.wp, [0.07, 0.09])
        a = np.exp( -(w - self.wp)**2 / (2 * sigma**2 * self.wp**2) )
        S = ((self.alpha * g**2 / w**5) * np.exp( -1.25 * (self.wp/w)**4 ) *
             self.gamma**a)
        return S

    @staticmethod
    def default_peak(Hs, Tp):
        """Default JONSWAP peak parameter, consistent with FAST"""
        TpOvrSqrtHs = Tp / np.sqrt(Hs)
        if TpOvrSqrtHs <= 3.6:
            return 5.0
        elif TpOvrSqrtHs >= 5.0:
            return 1.0
        else:
            return np.exp( 5.75 - 1.15*TpOvrSqrtHs )


class JONSWAP_Barltrop(object):
    """
    JONSWAP spectrum as quoted by Zhang2010 (p.54)
    """

    def __init__(self, Hs, Tz, gamma=None):
        """Three-parameter JONSWAP spectrum"""
        self.Hs = Hs
        self.Tz = Tz
        self.gamma = gamma

        self.kb = kb = 1.4085
        kp = 0.327 * np.exp(-0.315 * gamma) + 1.17
        kg = 1 - 0.285 * np.log(gamma)

        # Convert from zero-crossing period to peak frequency
        self.wp = 2*np.pi / (kp * Tz)

        # My own parameter for simplicity
        self.alpha = (kb**4 * Hs**2 * kg * 4 * pi**3) / (kp*Tz)**4

    def __call__(self, w):
        sigma = np.choose(w > self.wp, [0.07, 0.09])
        a = np.exp( -(w/self.wp - 1)**2 / (2 * sigma**2) )
        S = ((self.alpha / w**5) *
             np.exp( -(1/pi) / (w / self.wp / self.kb)**4 ) *
             self.gamma**a)
        return S
