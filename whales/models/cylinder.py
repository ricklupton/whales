"""
Models for floating vertical cylinders
"""

import numpy as np
from numpy import pi, diag, zeros, zeros_like, sinh, cosh
from scipy.special import jn, hankel2

from . import LinearSystem
from extra_special import jnd, hankel1d, hankel2d


def hydrostatics(draft, radius, mass):
    """Hydrostatic stiffness of vertical cylinder"""
    k_heave = 1025 * 9.81 * pi * radius**2
    k_pitch = (1025 * 9.81 * pi * radius**4 / 4) - ((draft/2) * mass * 9.81)
    K = diag([0, 0, k_heave, k_pitch, k_pitch, 0])
    return LinearSystem(zeros((6, 6)), zeros((6, 6)), K)


def Morison_added_mass(w, draft, radius, Cm=2.0):
    """Calculate added mass and damping from Morison strip model of
    uniform cylinder"""
    from whales.viscous_drag import ViscousDragModel
    Morison_model = ViscousDragModel({
        'inertia coefficient': Cm,
        'members': [{
            'end1': [0, 0, 0],
            'end2': [0, 0, -draft],
            'diameter': 2*radius,
            'strip width': 1.0,
        }]
    })
    A1 = Morison_model.Morison_added_mass()
    A = np.tile(A1, (len(w), 1, 1))
    return LinearSystem(A, zeros_like(A), zeros_like(A))


def first_order_excitation(k, draft, radius):
    """Returns F/(rho g a^2 A) -- from Drake"""
    ka = k * radius
    kd = k * draft

    # XXX check this!
    f1 = -1j * (jn(1, ka) - jnd(1, ka) * hankel2(1, ka) / hankel2d(1, ka))
    M = (-2 * pi / ka) * f1 * (1 - cosh(kd)) / (k * cosh(kd))
    F = (-2 * pi / ka) * f1 * sinh(kd) / cosh(kd)
    X = np.zeros(M.shape + (6,), dtype=np.complex)
    X[..., 0] = F
    X[..., 4] = M
    return X


def excitation_force(w, draft, radius):
    """Excitation force on cylinder using Drake's first_order_excitation"""
    k = w**2 / 9.81
    X = 1025 * 9.81 * radius**2 * first_order_excitation(k, draft, radius)
    return X


def mean_surge_force(w, depth, radius, RAOs, N=10):
    """Mean surge force on a surging and pitching cylinder
    From Drake2011.
    """

    g = 9.81
    k = w**2 / g     # XXX deep water
    ka = k * radius
    kd = k * depth

    # This is the part of the force which depends on the incoming waves
    hankel_term = lambda n, ka: 1 - ((hankel1d(n, ka) * hankel2d(n-1, ka)) /
                                     (hankel2d(n, ka) * hankel1d(n-1, ka)))
    terms = np.nansum([hankel_term(n, ka) for n in range(1, N)], axis=0)
    incoming_part = (1 + (2*kd/sinh(2*kd))) * terms

    # This is the part which depends on the first-order motion
    hankel_term2 = (hankel2d(0, ka) / hankel1d(0, ka) +
                    hankel2d(2, ka) / hankel1d(2, ka))
    motion_part = 2j * ((RAOs[:, 4] * (1 - cosh(kd)) / k +
                         RAOs[:, 0] * sinh(kd)) /
                        (cosh(kd) * hankel2d(1, ka)) * hankel_term2)

    return 0.5 / ka * np.real(incoming_part + motion_part)
