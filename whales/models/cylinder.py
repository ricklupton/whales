"""
Models for floating vertical cylinders
"""

import numpy as np
from numpy import pi, diag, zeros, zeros_like, sinh, cosh, newaxis
from scipy.special import jn, hankel1, hankel2

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


def first_order_excitation(k, draft, radius, water_depth):
    """Returns F/(rho g a^2 A) -- from Drake, but adapted by me to
    integrate down to the draft depth only (not to the seabed)"""
    ka = k * radius
    kd = k * draft
    kh = k * water_depth

    # XXX check this!
    f1 = -1j * (jn(1, ka) - jnd(1, ka) * hankel2(1, ka) / hankel2d(1, ka))
    #f1 = -1j * (jn(1, ka) - jnd(1, ka) * hankel1(1, ka) / hankel1d(1, ka))
    M = (kd*sinh(kh-kd) + cosh(kh-kd) - cosh(kh)) / (k * cosh(kh))
    F = (-sinh(kh-kd) + sinh(kh)) / cosh(kh)
    X = np.zeros(M.shape + (6,), dtype=np.complex)
    X[..., 0] = (-2*pi/ka) * f1 * F
    X[..., 4] = (-2*pi/ka) * f1 * M
    return X


def excitation_force(w, draft, radius, water_depth):
    """Excitation force on cylinder using Drake's first_order_excitation"""
    k = w**2 / 9.81
    ka = k * radius
    kd = k * draft
    kh = k * water_depth

    rho = 1025
    g = 9.81

    # XXX check this!
    f1 = -1j * (jn(1, ka) - jnd(1, ka) * hankel2(1, ka) / hankel2d(1, ka))
    #f1 = -1j * (jn(1, ka) - jnd(1, ka) * hankel1(1, ka) / hankel1d(1, ka))
    M = (kd*sinh(kh-kd) + cosh(kh-kd) - cosh(kh)) / (k**2 * cosh(kh))
    F = (-sinh(kh-kd) + sinh(kh)) / (k * cosh(kh))

    zs = zeros_like(F, dtype=np.complex)
    X = np.c_[F, zs, zs, zs, M, zs]
    X *= (-rho * g * pi * radius) * 2 * f1[:, newaxis]

    return X


def mean_surge_force(w, depth, radius, RAOs, N=10, deep_water=False):
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
    if deep_water:
        incoming_part = 1 * terms
    else:
        incoming_part = (1 + (2*kd/sinh(2*kd))) * terms

    # This is the part which depends on the first-order motion
    hankel_term2 = (hankel2d(0, ka) / hankel1d(0, ka) +
                    hankel2d(2, ka) / hankel1d(2, ka))
    if deep_water:
        motion_part = 2j * ((RAOs[:, 0] - (RAOs[:, 4] / k)) *
                            hankel_term2 / hankel2d(1, ka))
    else:
        motion_part = 2j * ((RAOs[:, 4] * (1 - cosh(kd)) / k +
                             RAOs[:, 0] * sinh(kd)) /
                            (cosh(kd) * hankel2d(1, ka)) * hankel_term2)

    return 0.5 / ka * np.real(incoming_part + motion_part)


def mean_surge_force2(w, depth, radius, RAOs, N=10, deep_water=False):
    """Mean surge force on a surging and pitching cylinder
    From Drake2011.
    """

    surge_rao = RAOs[:, 0] - depth*RAOs[:, 4]
    pitch_rao = RAOs[:, 4]

    g = 9.81
    k = w**2 / g     # XXX deep water
    ka = k * radius
    kd = k * depth

    # This is the part of the force which depends on the incoming waves
    hankel_term = lambda n, ka: 1 - ((hankel1d(n, ka) * hankel2d(n-1, ka)) /
                                     (hankel2d(n, ka) * hankel1d(n-1, ka)))
    terms = np.nansum([hankel_term(n, ka) for n in range(1, N)], axis=0)
    if deep_water:
        incoming_part = 1 * terms
    else:
        incoming_part = (1 + (2*kd/sinh(2*kd))) * terms

    # This is the part which depends on the first-order motion
    hankel_term2 = (hankel2d(0, ka) / hankel1d(0, ka) +
                    hankel2d(2, ka) / hankel1d(2, ka))
    if deep_water:
        motion_part = 2j * ((surge_rao + pitch_rao * depth * (1 - (1/kd))) *
                            hankel_term2 / hankel2d(1, ka))
    else:
        motion_part = 2j * ((pitch_rao * (1 - cosh(kd) + kd*sinh(kd)) +
                             surge_rao * k * sinh(kd)) /
                            (k * cosh(kd) * hankel2d(1, ka))
                            * hankel_term2)

    return 0.5 / ka * np.real(incoming_part + motion_part)
