"""
Wave spectra
"""

import numpy as np

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
