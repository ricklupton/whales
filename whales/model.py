"""
Input for floating wind turbine model
"""

import numpy as np
from numpy import linalg, newaxis
from scipy import interpolate, integrate
import h5py
import yaml

from whales.io import WAMITData, SecondOrderData
from whales.viscous_drag import ViscousDragModel
from whales.utils import skew, response_spectrum

from whales.structural_model import FloatingTurbineStructure
from whales.hydrodynamics import HydrodynamicsInfo


class FloatingTurbineModel(object):
    def __init__(self, config):
        # Build structural model
        self.structure = FloatingTurbineStructure(config['structure'])

        # Get hydrodynamic/static information
        self.hydro_info = HydrodynamicsInfo(config['hydrodynamics'])

        # Get mooring line behaviour
        l = config['mooring']['linear']
        self.mooring_static = np.asarray(l['static force'])
        self.mooring_stiffness = np.asarray(l['stiffness'])

        # Extra damping
        self.extra_damping = np.asarray(
            config['hydrodynamics'].get('extra damping', np.zeros(6)))
        if self.extra_damping.ndim == 1:
            self.extra_damping = np.diag(self.extra_damping)

        # Viscous drag model
        self.morison = ViscousDragModel(
            config['hydrodynamics']['Morison elements'])
        self.Bv = np.zeros((6, 6))
        self.Fvc = np.zeros(6)
        self.Fvv = np.zeros(6)

        # Wave-drift damping
        self.Bwd = np.zeros((6, 6))

    def calculate_viscous_effects(self, ws, S_wave):
        """
        Calculate viscous forces and damping with the given wave spectrum.

        Note that this can be iterative, as the transfer functions will be
        re-calculated using previously-calculated viscous effects.
        """
        H_wave = self.transfer_function_from_wave_elevation(ws, S_wave)
        Bv, Fvc, Fvv = self.morison.total_drag(ws, H_wave, S_wave)
        self.Bv, self.Fvc, self.Fvv = Bv, Fvc, Fvv

    def calculate_wave_drift_damping(self, ws, S_wave):
        """
        Calculate and save wave-drift damping matrix
        """
        self.Bwd = self.hydro_info.wave_drift_damping(ws, S_wave)

    def transfer_function(self, ws):
        """
        Linearise the structural model and return multi-dof transfer function
        for the structure, including mooring lines and hydrodynamics, at the
        given frequencies ``ws``
        """

        # Structural
        M_struct, B_struct, C_struct = self.structure.linearised_matrices()

        hydro = self.hydro_info
        C = hydro.C + self.C_grav + self.C_lines
        H = np.empty((len(ws), 6, 6), dtype=np.complex)

        for i, w in enumerate(ws):
            Mi = M + hydro.A(w)
            Bi = hydro.B(w) + self.Bv + self.Bwd + self.extra_damping

            H[i,:,:] = linalg.inv(-(w**2)*Mi + 1j*w*Bi + C)
        return H

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
