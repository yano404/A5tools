"""
functions to convert the lab system to the center-of-mass system
"""

import numpy as np
import uncertainties
from uncertainties import umath
import uncertainties.unumpy as unp
from .constants import *


def theta(theta_lab, n):
    """
    function to convert θ in the lab system to θ in the center-of-mass system

    Arguments
    ---------
    theta_lab : scattering angle in the lab system [rad]
    n         : A_t / A_i; A_t, A_i means mass number of target particle or incident particle

    Return
    ------
    theta_cm  : scattering angle in the center-of-mass system [rad]

    Notice
    ------
    This function do not consider relativity
    """
    if isinstance(theta_lab, uncertainties.core.AffineScalarFunc):
        coslab2 = umath.pow(umath.cos(theta_lab), 2.0)
        coscm = (coslab2 - 1.0) / n \
            + umath.sqrt((1.0 - 1.0 / n**2.0) * coslab2 +
                         umath.pow(coslab2 / n, 2.0))
        return np.sign(theta_lab) * umath.acos(coscm)
    elif isinstance(theta_lab, np.ndarray) and isinstance(theta_lab[0], uncertainties.core.AffineScalarFunc):
        coslab2 = unp.pow(unp.cos(theta_lab), 2.0)
        coscm = (coslab2 - 1.0) / n \
            + unp.sqrt((1.0 - 1.0 / n**2.0) * coslab2 +
                       unp.pow(coslab2 / n, 2.0))
        return np.sign(theta_lab) * unp.arccos(coscm)
    else:
        coslab2 = np.power(np.cos(theta_lab), 2.0)
        coscm = (coslab2 - 1.0) / n \
            + np.sqrt((1.0 - 1.0 / n**2.0) * coslab2 +
                      np.power(coslab2 / n, 2.0))
        return np.sign(theta_lab) * np.arccos(coscm)


def dOmega(theta_lab, n):
    """
    function to find dΩlab/dΩcm

    Arguments
    ---------
    theta_lab : scattering angle in the lab system [rad]
    n         : A_t / A_i; A_t, A_i means mass number of target particle or incident particle

    Return
    ------
      dΩlab
     -------- : factor to convert differential cross-section in the lab system to differential cross-section in the center-of-mass system
      dΩcm

    Notice
    ------
    This function do not consider relativity
    """
    if isinstance(theta_lab, uncertainties.core.AffineScalarFunc):
        return umath.pow(
            2.0 * umath.cos(theta_lab) / n
            + (1.0 + umath.cos(2.0 * theta_lab) / (n**2.0))
            / umath.sqrt(1.0 - umath.pow(umath.sin(theta_lab) / n, 2.0)), -1.0
        )
    elif isinstance(theta_lab, np.ndarray) and isinstance(theta_lab[0], uncertainties.core.AffineScalarFunc):
        return unp.pow(
            2.0 * unp.cos(theta_lab) / n
            + (1.0 + unp.cos(2.0 * theta_lab) / (n**2.0))
            / unp.sqrt(1.0 - unp.pow(unp.sin(theta_lab) / n, 2.0)), -1.0
        )
    else:
        return np.power(
            2.0 * np.cos(theta_lab) / n
            + (1.0 + np.cos(2.0 * theta_lab) / (n**2.0))
            / np.sqrt(1.0 - np.power(np.sin(theta_lab) / n, 2.0)), -1.0
        )


def kinetic_energy(Tlab, Ai, At):
    """
    function to convert kinetic energy in the lab system to kinetic energy in the center-of-mass system

    Arguments
    ---------
    Tlab : kinetic energy in the lab system [MeV]
    Ai   : mass number of incident particle
    At   : mass number of target particle

    Return
    ------
    Tcm  : kinetic energy in the lab system [MeV]
    """
    # rest energy: mc^2
    mic2 = Ai * AMU  # [MeV]
    mtc2 = At * AMU  # [MeV]
    # total energy of incident particle in the lab system
    Eilab = Tlab + mic2
    # total energy in the center-of-mass system
    Ecm = np.sqrt(2.0 * Eilab * mtc2 + mic2**2.0 + mtc2**2.0)
    # kinetic energy in the center-of-mass system
    return Ecm - mic2 - mtc2
