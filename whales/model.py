"""
Input for floating wind turbine model
"""

import numpy as np
from numpy import linalg, newaxis
from scipy import interpolate, integrate
import h5py
import yaml

from whales import WAMIT
from whales.viscous_drag import ViscousDragModel
from whales.utils import skew, shift, response_spectrum

g = 9.81

class Body(object):
    def __init__(self, name, position, mass, inertia=None):
        if inertia is None:
            inertia = np.zeros((3,3))

        position = np.asarray(position)
        inertia = np.asarray(inertia)
        if inertia.ndim == 1:
            inertia = np.diag(inertia)

        self.name = name
        self.position = position
        self.mass = mass
        self.inertia = inertia

    @property
    def mass_matrix(self):
        xs = skew(self.position)
        m,I = self.mass, self.inertia
        return np.r_[ np.c_[ m*np.eye(3), -m*xs                ] ,
                      np.c_[ m*xs,         I - m*np.dot(xs, xs)] ]

    def __str__(self):
        return 'Body "%s" at %s' % (self.name, self.position)
    def __repr__(self):
        return '<%s>' % self

class HydrodynamicsInfo(object):
    def __init__(self, config):
        self.wamit = WAMIT.HydroData(config['WAMIT path'])

        # make smallest period very small instead of zero
        self.wamit.periods[self.wamit.periods < 1e-10] = 1e-10

        # sort by frequency not period
        w = 2*np.pi/self.wamit.periods
        idx = np.argsort(w)
        self.w = w[idx]
        self.k = self.w**2 / g

        self.wamit.A = self.wamit.A[idx]
        # Damping goes to nan as w -> 0
        self.wamit.B = np.nan_to_num(self.wamit.B[idx])
        self.wamit.C = self.wamit.C
        self.wamit.X = self.wamit.X[idx]

        # Make interpolators
        self.A = interpolate.interp1d(self.w, self.wamit.A, axis=0)
        self.B = interpolate.interp1d(self.w, self.wamit.B, axis=0)
        self.C = self.wamit.C
        self.X = interpolate.interp1d(self.w, self.wamit.X, axis=0)

        # QTFs
        self.Tc = None
        if 'QTF path' in config:
            with h5py.File(config['QTF path']) as f:
                w_Tc = f['diff/diag/w'][...]
                Tc = f['diff/diag/Tc'][...]
            # De-normalise
            Tc *= 1025 * g
            if w_Tc[0] != 0:
                print "Assuming low-frequency asymptote of zero"
                w_Tc = np.r_[ 0, w_Tc ]
                Tc = np.r_[ np.zeros((1,6)), Tc ]
                #Tc = np.r_[ Tc[0:1,:], Tc ]
            self.Tc = interpolate.interp1d(w_Tc, Tc, axis=0)

    def wave_drift_damping(self, w, S_wave):
        """Calculate wave-drift damping coefficient"""
        Tc = self.Tc(w)
        # Only include surge and sway
        # XXX is this right? do I need to treat sway differently?
        dTc_dw  = np.gradient(Tc, w[1] - w[0])[0] # gradient along axis 0

        integrand = S_wave[:,newaxis] * (4 * w[:,newaxis] * Tc +
                                         w[:,newaxis]**2 * dTc_dw) / g
        B12 = integrate.simps(integrand, w, axis=0)

        B = np.zeros((6,6))
        B[0,0] = B12[0]
        B[1,1] = B12[1]
        return B

    def first_order_force_spectrum(self, w, S_wave, heading=0):
        ih = np.nonzero(self.wamit.headings == heading)[0][0]
        # Interpolate complex wave excitation force
        X = self.X(w)[:,ih,:]
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
            SS[k,:] *= shift(S_wave, k, fill=0)

        # Next use the Newman approximation to make a similar matrix TTc
        Tc = self.Tc(w) # interpolate -> (len(w), 6) matrix
        TTc = np.tile(Tc, (len(w), 1, 1)) # copy downwards new 0 axis
        for k in range(len(w)):
            for i in range(6):
                TTc[k,:,i] += shift(Tc[:,i], k)
        TTc /= 2 # average

        # Now loop through each DOF to make the 2nd order spectrum
        S2 = np.zeros((len(w), 6, 6))
        for i in range(6):
            for j in range(6):
                integrand = SS * TTc[:,:,i] * TTc[:,:,j]
                S2[:,i,j] = 8 * integrate.simps(integrand, w, axis=1)
        return S2

class FloatingTurbineModel(object):
    def __init__(self, config):
        s = config['structure']
        self.platform = Body('platform', s['platform']['CoM'],
                             s['platform']['mass'], s['platform']['inertia'])
        self.tower = Body('tower', s['tower']['CoM'],
                          s['tower']['mass'], s['tower']['inertia'])
        self.nacelle = Body('nacelle', np.array([0, 0, s['nacelle']['height']]),
                            s['nacelle']['mass'])
        self.rotor = Body('rotor', np.array([0, 0, s['nacelle']['height']]),
                          s['rotor']['mass'], s['rotor']['inertia'])
        self.bodies = (self.platform, self.tower, self.nacelle, self.rotor)

        self.hydro_info = HydrodynamicsInfo(config['hydrodynamics'])

        l = config['mooring']['linear']
        self.mooring_static = np.asarray(l['static force'])
        self.mooring_stiffness = np.asarray(l['stiffness'])
        self.C_lines = self.mooring_stiffness

        self.extra_damping = np.asarray(config['hydrodynamics']
                                        .get('extra damping', [0]*6))
        if self.extra_damping.ndim == 1:
            self.extra_damping = np.diag(self.extra_damping)

        self.morison = ViscousDragModel(config['hydrodynamics']['Morison elements'])

    #### Inertial properties
    @property
    def mass(self):
        return (self.platform.mass + self.tower.mass +
                self.nacelle.mass + self.rotor.mass)

    @property
    def mass_matrix(self):
        return sum(xx.mass_matrix for xx in self.bodies)
    M = mass_matrix

    @property
    def added_mass_matrix(self):
        return self.hydro_info.A
    A = added_mass_matrix

    #### Damping properties
    @property
    def hydro_damping_matrix(self):
        return self.hydro_info.B
    B = hydro_damping_matrix

    #### Stiffness properties
    @property
    def hydrostatic_stiffness(self):
        return self.hydro_info.C
    C_hydro = hydrostatic_stiffness

    @property
    def gravitational_stiffness(self):
        # XXX is this general?
        C = np.zeros((6,6))
        C[3,3] = C[4,4] = -g * sum(b.position[2]*b.mass for b in self.bodies)
        return C
    C_grav = gravitational_stiffness

    #C_lines = mooring_stiffness

    #### Response
    def transfer_function(self, ws, added_mass=True, radiation_damping=True,
                          drift_damping=True, extra_damping=True,
                          viscous_damping=None, S_wave=None):

        if viscous_damping is None:
            viscous_damping = np.zeros((6,6))
        if drift_damping:
            if S_wave is None:
                raise ValueError("Need wave spectrum to calculate wave drift damping")
            wave_drift_damping = self.hydro_info.wave_drift_damping(ws, S_wave)
        else:
            wave_drift_damping = np.zeros((6,6))

        M, A, B = self.M, self.A, self.B
        C = self.C_hydro + self.C_grav + self.C_lines
        H = np.empty((len(ws), 6, 6), dtype=np.complex)

        print wave_drift_damping[0,0], viscous_damping[0,0], self.extra_damping[0,0]

        for i,w in enumerate(ws):
            Mi = M + (added_mass * A(w))
            Bi = ((radiation_damping * B(w)) + viscous_damping +
                  wave_drift_damping + (extra_damping * self.extra_damping))
            H[i,:,:] = linalg.inv(-(w**2)*Mi + 1j*w*Bi + C)
        return H

    def transfer_function_from_wave_elevation(self, ws, heading=0,
                                              viscous_drag=None, **tf_options):
        ih = np.nonzero(self.hydro_info.wamit.headings == heading)[0][0]
        H = self.transfer_function(ws, **tf_options)
        X = self.hydro_info.X(ws) # interpolate
        X = X[:,ih,:]

        # Add in the viscous drag force due to waves (the constant
        # current part is not included here)
        if viscous_drag is not None:
            X += viscous_drag

        H_wave = np.einsum('wij,wj->wi', H, X)
        return H_wave

    def response_spectrum(self, ws, force_spectrum, **options):
        """Calculate the response spectrum using the eqn from Naess2013 (9.79):
        $$ S_{x_i x_j} = \sum_r \sum_s H_{ir}(\omega) H_{js}^{*}(\omega)
        S_{F_r F_s}(\omega) $$
        """
        H = self.transfer_function(ws, **options)
        Sx = np.einsum('wir,wjs,wrs->wij', H, H.conj(), force_spectrum)
        return Sx

    @classmethod
    def from_yaml(cls, filename):
        with open(filename) as f:
            config = yaml.safe_load(f)
        return cls(config)
