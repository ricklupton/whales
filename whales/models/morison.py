"""
Models made up of Morison elements
"""

import numpy as np
from numpy import zeros_like

from . import LinearSystem
from whales.viscous_drag import ViscousDragModel


def excitation_force(w, z1, z2, diameter, Cm=2):
    """Calculate the excitation force. Specific for now to vertical
    cylinders of varying diameter.

    ``z1``, ``z2``: coordinates of ends of element
    ``diameter``: array (2 x Npoints) diameter at each z value
    """

    config = {
        'end1': [0, 0, z1],
        'end2': [0, 0, z2],
        'diameter': diameter,
        'strip width': 1.0,
    }
    Morison_model = ViscousDragModel({
        'inertia coefficient': Cm,
        'members': [config],
    })
    X = Morison_model.Morison_inertial_force(w)
    return X


def added_mass(w, z1, z2, diameter, Cm=2):
    """Calculate the added mass/damping. Specific for now to vertical
    cylinders of varying diameter.

    ``z1``, ``z2``: coordinates of ends of element
    ``diameter``: array (2 x Npoints) diameter at each z value
    """

    config = {
        'end1': [0, 0, z1],
        'end2': [0, 0, z2],
        'diameter': diameter,
        'strip width': 1.0,
    }
    Morison_model = ViscousDragModel({
        'inertia coefficient': Cm,
        'members': [config],
    })
    A1 = Morison_model.Morison_added_mass()
    A = np.tile(A1, (len(w), 1, 1))
    return LinearSystem(A, zeros_like(A), zeros_like(A))
