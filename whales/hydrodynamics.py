"""
Hydrodynamic calculations -- force spectra etc
"""

import numpy as np
from numpy import newaxis
from scipy import integrate, constants
from scipy.interpolate import interp1d

from whales.utils import shift, response_spectrum


class HydrodynamicsInfo(object):
    def __init__(self, first_order_data, second_order_data=None):
        self.first_order_data = first_order_data
        self.second_order_data = second_order_data

        # Make interpolators
        self.A = interp1d(first_order_data.w, first_order_data.A, axis=0)
        self.B = interp1d(first_order_data.w, first_order_data.B, axis=0)
        self.C = first_order_data.C

        # QTFs
        if second_order_data is not None:
            self.Tc = interp1d(second_order_data.w,
                               second_order_data.Tc, axis=0)
        else:
            self.Tc = None

    def X(self, w, heading):
        Xh = interp1d(self.first_order_data.headings,
                      self.first_order_data.X, axis=1)
        Xw = interp1d(self.first_order_data.w, Xh(heading), axis=0)
        return Xw(w)

    def wave_drift_damping(self, w, S_wave):
        """Calculate wave-drift damping coefficient"""
        Tc = self.Tc(w)
        # Only include surge and sway
        # XXX is this right? do I need to treat sway differently?
        dTc_dw = np.gradient(Tc, w[1] - w[0])[0]  # gradient along axis 0

        integrand = (S_wave[:, newaxis] *
                     (4 * w[:, newaxis] * Tc + w[:, newaxis]**2 * dTc_dw) /
                     constants.g)
        B12 = integrate.simps(integrand, w, axis=0)

        B = np.zeros((6, 6))
        B[0, 0] = B12[0]
        B[1, 1] = B12[1]
        return B

    def first_order_force_spectrum(self, w, S_wave, heading=0):
        # Interpolate complex wave excitation force
        X = self.X(w, heading)
        return response_spectrum(X, S_wave)

    def second_order_force_spectrum(self, w, S_wave, heading=0):
        """Calculate the 2nd order force spectrum defined by
        $$ S_i^{(2)}(\omega) =
        8 \int S_{\eta\eta}(\mu) S_{\eta\eta}(\mu + \omega) \,
        \left| T_i^{\mathrm{c}}(\mu, \mu + \omega) \right|^2 \mathrm{d}\mu $$
        but extended to give the cross-spectrum S_ij
        """

        if heading != 0:
            raise RuntimeError("No QTF data for headings other than 0")

        # First make the matrix SS_kl, where l is the freq and k is the offset
        #  SS_kl = S(w_l) * S(w_l + w_k)
        SS = np.tile(S_wave, (len(w), 1))
        for k in range(len(w)):
            SS[k, :] *= shift(S_wave, k, fill=0)

        # Next use the Newman approximation to make a similar matrix TTc
        Tc = self.Tc(w)  # interpolate -> (len(w), 6) matrix
        TTc = np.tile(Tc, (len(w), 1, 1))  # copy downwards new 0 axis
        for k in range(len(w)):
            for i in range(6):
                TTc[k, :, i] += shift(Tc[:, i], k)
        TTc /= 2  # average

        # Now loop through each DOF to make the 2nd order spectrum
        S2 = np.zeros((len(w), 6, 6))
        for i in range(6):
            for j in range(6):
                integrand = SS * TTc[:, :, i] * TTc[:, :, j]
                S2[:, i, j] = 8 * integrate.simps(integrand, w, axis=1)
        return S2
