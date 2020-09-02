"""
utility functions
"""

import numpy as np
from .constants import *


def gauss_int(con, sigma):
    """
    Gaussian Integral
    sigma must be greater than 0
    """
    return np.sqrt(2.0 * np.pi) * con * sigma


def bethe(z, a, Z, A, Ei):
    """
    function to calc -dE/d(ρx) [MeV mb / g]

    Arguments
    ---------
    z: charge of incident particle in units of e [Int]
    a: Atomic Weight of incident particle [Int]
    Z: atomic number of absorbing material [Int]
    A: Atomic Weight of absorbing material [Int]
    Ei: Energy of incident particle [MeV]

    Return
    ------
    -dE/d(ρx) [MeV mb / g]
    """
    # Constant part of Bethe-Bloch equation
    const = 0.1535E+27  # [MeV mb / g]
    # calc gamma
    gamma = (Ei / AMU + a) / a
    # calc beta
    beta = np.sqrt(1.0 - 1.0 / gamma**2.0)
    # eta
    eta = beta * gamma
    # wmax
    s = MEC2 / (a * AMU)
    wmax = (2.0 * MEC2 * eta**2.0) / \
        (1.0 + 2.0 * s * np.sqrt(1.0 + eta**2.0) + s**2.0)
    # I
    if Z < 13:
        I = (12 * Z + 7) * 1E-6  # [MeV]
    else:
        I = (9.76 * Z + 58.8 * np.pow(Z, -0.19)) * 1E-6  # [MeV]
    # return -dE/d(ρx)
    return - const * (Z / A) \
        * (z**2.0 / beta**2.0) \
        * (np.log(2.0 * MEC2 * gamma**2.0 * beta**2.0 * wmax / I**2.0) - 2.0 * beta**2.0)
