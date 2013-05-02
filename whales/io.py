"""
Tools for loading WAMIT output data
"""

import numpy as np
from numpy import newaxis, pi, isinf
import h5py
from scipy import constants

class WAMITData(object):
    def __init__(self, path, skiprows=0, water_density=1e3):
        """Load WAMIT results data from the run ``path``:
         - The hydrodynamic added-mass matrix ``A``
         - The hydrodynamic radiation damping matrix ``B``
         - The hydrostatic restoring force matrix ``C``
         - The wave excitation forces ``X``

        ``A`` and ``B`` are frequency-dependent matrices of shape
        (frequency, 6, 6). ``C`` is a constant matrix of shape (6, 6).
        ``X`` is a frequency- and heading-dependent vector of shape
        (frequency, heading, 6).

        These quantities are denormalised using the supplied value
        for water density, and standard gravitational acceleration.
        """
        self.path = path
        self.water_density = water_density

        # Radiation data:
        #      A: Added mass
        #      B: Radiation damping
        #      T: Wave period (s)
        #   i, j: motion indices
        rad = np.genfromtxt(self.path + '.1', skiprows=skiprows,
                            names=('T', 'i', 'j', 'A', 'B'),
                            delimiter=(14, 6, 6, 14, 14))

        # Wave excitation data
        ex = np.genfromtxt(self.path + '.3', skiprows=skiprows,
                           names=('T', 'head', 'i', 'mod', 'pha', 're', 'im'))

        # Restoring force data
        rest = np.genfromtxt(self.path + '.hst', skiprows=skiprows,
                             names=('i', 'j', 'C'), comments='!')

        # Get set of periods and headings
        period_list = sorted(set(rad['T']) | set(ex['T']))
        heading_list = sorted(set(ex['head']))
        periods = np.array(period_list)
        headings = np.array(heading_list)

        with np.errstate(divide='ignore'):
            w = 2 * np.pi / periods
        w[periods == -1] = 0
        w[periods == 0] = np.inf

        # Rearrange the data
        self.A = np.zeros((len(periods), 6, 6))
        self.B = np.zeros((len(periods), 6, 6))
        for row in rad:
            ip = period_list.index(row['T'])
            self.A[ip, row['i']-1, row['j']-1] = row['A']
            self.B[ip, row['i']-1, row['j']-1] = row['B']

        self.X = np.zeros((len(periods), len(headings), 6), dtype=np.complex)
        for row in ex:
            ip = period_list.index(row['T'])
            ih = heading_list.index(row['head'])
            self.X[ip, ih, row['i']-1] = row['re'] + 1j*row['im']

        self.C = np.zeros((6, 6))
        for row in rest:
            self.C[row['i']-1, row['j']-1] = row['C']

        # Fix missing values for damping at zero and infinite frequency
        self.B = np.nan_to_num(self.B)

        # Re-dimensionalise (note damping should go to zero as w -> 0)
        self.A *= water_density
        self.B[~isinf(w)] *= water_density * w[~isinf(w)][:, newaxis, newaxis]
        self.B[isinf(w)] = 0
        self.C *= water_density * constants.g
        self.X *= water_density * constants.g

        # Re-arrange into frequency order (not period order)
        idx = np.argsort(w)
        self.w = w[idx]
        self.headings = headings
        self.A = self.A[idx]
        self.B = self.B[idx]
        self.X = self.X[idx]


class SecondOrderData(object):
    def __init__(self, path, water_density=1e3):
        """
        Load pre-processed QTF diagonal values from HDF5 file
        """
        self.path = path
        with h5py.File(path) as f:
            w = f['diff/diag/w'][...]
            Tc = f['diff/diag/Tc'][...]

        # De-normalise
        Tc *= water_density * constants.g
        if w[0] != 0:
            print "Assuming low-frequency asymptote of zero"
            w = np.r_[0, w]
            Tc = np.r_[np.zeros((1, 6)), Tc]

        self.w = w
        self.Tc = Tc
