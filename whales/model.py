"""
Input for floating wind turbine model
"""

import numpy as np
from numpy import newaxis, zeros_like, eye, r_, c_
from scipy import interpolate, integrate, linalg
import h5py
import yaml

from whales.io import WAMITData, SecondOrderData
from whales.viscous_drag import ViscousDragModel
from whales.utils import response_spectrum
from whales.structural_model import FloatingTurbineStructure
from whales.hydrodynamics import HydrodynamicsInfo


class FloatingTurbineModel(object):
    def __init__(self, config, frequency_values):
        self.config = config
        self.w = frequency_values

        # Build structural model
        self.structure = FloatingTurbineStructure(config['structure'])

        # Get hydrodynamic/static information
        h = config['hydrodynamics']
        rho = config['constants']['water density']
        first_order = WAMITData(h['WAMIT path'], water_density=rho)
        if 'QTF path' in h:
            second_order = SecondOrderData(h['QTF path'], water_density=rho)
        else:
            second_order = None
        self.hydro_info = HydrodynamicsInfo(first_order, second_order)

        # Get mooring line behaviour
        l = config['mooring']['linear']
        self.mooring_static = np.asarray(l['static force'])
        self.mooring_stiffness = np.asarray(l['stiffness'])

        # Extra damping
        self.B_extra = np.asarray(h.get('extra damping', np.zeros(6)))
        if self.B_extra.ndim == 1:
            self.B_extra = np.diag(self.B_extra)

        # Viscous drag model
        self.morison = ViscousDragModel(h['Morison elements'])
        self.Bv = np.zeros((6, 6))
        self.Fvc = np.zeros(6)
        self.Fvv = np.zeros((len(self.w), 6))

        # Wave-drift damping
        self.Bwd = np.zeros((6, 6))

    def calculate_viscous_effects(self, S_wave):
        """
        Calculate viscous forces and damping with the given wave spectrum.

        Note that this can be iterative, as the transfer functions will be
        re-calculated using previously-calculated viscous effects.
        """
        assert len(S_wave) == len(self.w)
        H_wave = self.transfer_function_from_wave_elevation()
        self.Bv, self.Fvc, self.Fvv = self.morison.total_drag(self.w, H_wave,
                                                              S_wave)

        # Fvv is the actual force/unit wave height at each wave
        # frequency -- to get the spectrum, it's like the first-order
        # wave force spectrum.
        self.viscous_force_spectrum = response_spectrum(self.Fvv, S_wave)

    def calculate_wave_drift_damping(self, S_wave):
        """
        Calculate and save wave-drift damping matrix
        """
        assert len(S_wave) == len(self.w)
        self.Bwd = self.hydro_info.wave_drift_damping(self.w, S_wave)

    def linearised_matrices(self, w, **kwargs):
        """
        Linearise the structural model and add in the hydrodynamic
        added-mass and damping for the given frequency ``w``, as well
        as the hydrostatic stiffness, wave-drift damping and viscous damping.
        """
        # Set default perturbation
        kwargs.setdefault('perturbation', 1e-4)

        # Structural - includes gravitational stiffness
        M, B, C, = self.structure.linearised_matrices(**kwargs)
        M[:6, :6] += self.hydro_info.A(w)
        B[:6, :6] += self.hydro_info.B(w) + self.Bv + self.Bwd + self.B_extra
        C[:6, :6] += self.hydro_info.C + self.mooring_stiffness
        return M, B, C

    def coupled_modes(self, w, ignore_damping=False, **kwargs):
        M, B, C = self.linearised_matrices(w, **kwargs)
        if ignore_damping:
            wn, vn = linalg.eig(C, M)
        else:
            AA = r_[c_[zeros_like(C), C], c_[C, B]]
            BB = r_[c_[C, zeros_like(C)], c_[zeros_like(C), -M]]
            wn, vn = linalg.eig(AA, BB)
            order = np.argsort(abs(wn))
            wn = abs(wn[order])
            # Mode shapes are the first half of the rows; the second
            # half of the rows should be same multiplied by eigenvalues.
            vn = vn[:M.shape[0], order]

            # We expect all the modes to be complex conjugate; return
            # every other one.
            # First: make sure all are the same sign
            norm_vn = vn / vn[np.argmax(abs(vn), axis=0), range(vn.shape[1])]
            assert (np.allclose(wn[::2], wn[1::2], rtol=1e-4) and
                    np.allclose(norm_vn[:, ::2], norm_vn[:, 1::2].conj(), atol=1e-2)), \
                "Expect conjugate modes"
            wn = wn[::2]
            vn = norm_vn[:, ::2]
        return wn, vn

    def transfer_function(self, rotor_speed=None):
        """
        Linearise the structural model and return multi-dof transfer function
        for the structure, including mooring lines and hydrodynamics, at the
        given frequencies ``ws``
        """

        # Structural - includes gravitational stiffness
        if rotor_speed is not None:
            M_struct, B_struct, C_struct = self.structure.linearised_matrices(
                zd0={'shaft': [rotor_speed]}, mbc=True)
        else:
            M_struct, B_struct, C_struct = self.structure.linearised_matrices()

        # Full matrices to be assembled at each frequency
        Mi = zeros_like(M_struct)
        Bi = zeros_like(B_struct)
        Ci = zeros_like(C_struct)

        # Stiffness is constant -- add hydrostatics and mooring lines
        hh = self.hydro_info  # shorthand
        Ci[:, :] = C_struct
        Ci[:6, :6] += hh.C + self.mooring_stiffness

        # Calculate transfer function at each frequency
        H = np.empty((len(self.w),) + M_struct.shape, dtype=np.complex)
        for i, w in enumerate(self.w):
            Mi[:, :] = M_struct
            Bi[:, :] = B_struct
            Mi[:6, :6] += hh.A(w)  # add rigid-body parts
            Bi[:6, :6] += hh.B(w) + self.Bv + self.Bwd + self.B_extra
            H[i, :, :] = linalg.inv(-(w**2)*Mi + 1j*w*Bi + Ci)
        return H

    def transfer_function_from_wave_elevation(self, heading=0):
        """
        Calculate the transfer function from wave elevation to response,
        similar to ``transfer_function(ws)`` but including the wave excitation
        force ``X``.
        """
        H = self.transfer_function()
        X = self.hydro_info.X(self.w, heading)  # interpolate

        # Add in the viscous drag force due to waves
        X += self.Fvv
        X[self.w == 0] += self.Fvc  # constant drag force

        # Multiply transfer functions to get overall transfer function
        # (H can be larger than X if the model is flexible)
        H_wave = np.einsum('wij,wj->wi', H[:, :, :6], X)
        return H_wave

    def response_spectrum(self, S_wave, second_order=True, viscous=True):
        """Convenience method which calculates the response spectrum
        """
        S1 = self.hydro_info.first_order_force_spectrum(self.w, S_wave)
        if second_order:
            S2 = self.hydro_info.second_order_force_spectrum(self.w, S_wave)
        else:
            S2 = zeros_like(S1)
        if viscous:
            self.calculate_viscous_effects(S_wave)
            Sv = self.viscous_force_spectrum
        else:
            Sv = zeros_like(S1)
        self.calculate_wave_drift_damping(S_wave)

        # XXX this doesn't work well if second_order or viscous is set
        # to False -- transfer_function() will use previously-calculated values

        H = self.transfer_function()
        SF = S1 + S2 + Sv
        Sx = response_spectrum(H[:, :, :6], SF)
        return Sx

    @classmethod
    def from_yaml(cls, filename, freq):
        with open(filename) as f:
            config = yaml.safe_load(f)
        return cls(config, freq)
