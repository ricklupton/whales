"""
Tools for loading WAMIT output data
"""

import numpy as np

gravity = 9.81

class HydroData(object):
    def __init__(self, path, skiprows=0, water_density=1e3):
        self.path = path
        self.water_density = water_density

        # Radiation data:
        #      A: Added mass
        #      B: Radiation damping
        #      T: Wave period (s)
        #   i, j: motion indices
        rad = np.genfromtxt(self.path + '.1', skiprows=skiprows,
                            names=('T','i','j','A','B'),
                            delimiter=(14, 6, 6, 14, 14))

        # Wave excitation data
        ex = np.genfromtxt(self.path + '.3', skiprows=skiprows,
                           names=('T','head','i','mod','pha','re','im'))

        # Restoring force data
        rest = np.genfromtxt(self.path + '.hst', skiprows=skiprows,
                             names=('i','j','C'), comments='!')

        # Get set of periods and headings
        periods = sorted(set(rad['T']) | set(ex['T']))
        headings = sorted(set(ex['head']))
        self.periods = np.array(periods)
        self.periods[self.periods == -1] = np.inf
        self.headings = np.array(headings)

        # Rearrange the data
        self.A = np.zeros((len(periods), 6, 6))
        self.B = np.zeros((len(periods), 6, 6))
        for row in rad:
            ip = periods.index(row['T'])
            self.A[ip, row['i']-1, row['j']-1] = row['A']
            self.B[ip, row['i']-1, row['j']-1] = row['B']

        self.X = np.zeros((len(periods), len(headings), 6), dtype=np.complex)
        for row in ex:
            ip = periods.index(row['T'])
            ih = headings.index(row['head'])
            self.X[ip, ih, row['i']-1] = row['re'] + 1j*row['im']

        self.C = np.zeros((6,6))
        for row in rest:
            self.C[row['i']-1, row['j']-1] = row['C']

        # Re-dimensionalise
        self.A *= water_density
        self.B *= water_density * 2*np.pi / self.periods[:,np.newaxis,np.newaxis]
        self.C *= water_density * gravity
        self.X *= water_density * gravity
