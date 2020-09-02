"""
functions to convert the center-of-mass system to the lab system
"""

import numpy as np
import uncertainties
from uncertainties import umath
import uncertainties.unumpy as unp
from .constants import *


def theta(theta_cm, n):
    """
    function to convert θ in the center-of-mass system to θ in the lab system

    Arguments
    ---------
    theta_cm : scattering angle in the center-of-mass system [rad]
    n        : A_t / A_i; A_t, A_i means mass number of target particle or incident particle

    Return
    ------
    theta_lab : scattering angle in the lab system [rad]

    Notice
    ------
    This function does not consider relativity
    """
    if isinstance(theta_cm, uncertainties.core.Variable):
        return umath.arctan(umath.sin(theta_cm) / (1.0 / n + umath.cos(theta_cm)))
    elif isinstance(theta_cm, np.ndarray) and isinstance(theta_cm[0], uncertainties.core.Variable):
        return unp.arctan(unp.sin(theta_cm) / (1.0 / n + unp.cos(theta_cm)))
    else:
        return np.arctan(np.sin(theta_cm) / (1.0 / n + np.cos(theta_cm)))


def dOmega(theta_cm, n):
    """
    function to find dΩcm/dΩlab

    Arguments
    ---------
    theta_cm : scattering angle in the center-of-mass system [rad]
    n        : A_t / A_i; A_t, A_i means mass number of target particle or incident particle

    Return
    ------
      dΩcm
     -------- : factor to convert differential cross-section in the center-of-mass system to one in the lab system
      dΩlab

    Notice
    ------
    This function does not consider relativity
    """
    if isinstance(theta_cm, uncertainties.core.Variable):
        return umath.pow(1.0 + 2.0 * umath.cos(theta_cm) / n + 1.0 / n**2.0, 3 / 2) \
            / (1.0 + umath.cos(theta_cm) / n)
    elif isinstance(theta_cm, np.ndarray) and isinstance(theta_cm, uncertainties.core.Variable):
        return unp.pow(1.0 + 2.0 * unp.cos(theta_cm) / n + 1.0 / n**2.0, 3 / 2) \
            / (1.0 + unp.cos(theta_cm) / n)
    else:
        return np.power(1.0 + 2.0 * np.cos(theta_cm) / n + 1.0 / n**2.0, 3 / 2) \
            / (1.0 + np.cos(theta_cm) / n)


def kinetic_energy(Tcm, Ai, At):
    """
    function to convert kinetic energy in the center-of-mass system to kinetic energy in the lab system

    Arguments
    ---------
    Tcm  : kinetic energy in the center-of-mass system [MeV]
    Ai   : mass number of incident particle
    At   : mass number of target particle

    Return
    ------
    Tlab : kinetic energy in the lab system [MeV]
    """
    # rest energy: mc^2
    mic2 = Ai * AMU  # [MeV]
    mtc2 = At * AMU  # [MeV]
    # total energy in the center-of-mass system
    Ecm = Tcm + mic2 + mtc2
    # total energy of incident particle in the lab system
    Eilab = (Ecm**2.0 - mic2**2.0 - mtc2**2.0) / (2.0 * mtc2)
    # kinetic energy of incident particle in the lab system
    return Eilab - mic2
