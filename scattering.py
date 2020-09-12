"""
functions to find theoretical value of differential cross-section
"""

import numpy as np
import uncertainties
from uncertainties import umath
import uncertainties.unumpy as unp
from .constants import *
from . import cm2lab
from . import lab2cm


def rutherford(theta, T, Zi, Zt, which="inc"):
    """
    Rutherford Scattering in the center-of-mass system

    Arguments
    ---------
    theta : scattering angle in the center-of-mass system
    T     : kinetic energy of incident particle in the center-of-mass system
    Zi    : atomic number of incident particle; charge in units of e
    Zt    : atomic number of target particle; charge in units of e
    which : if which="inc", calc dσ/dΩ(θ) of incident particle.
            if which="tar", calc dσ/dΩ(θ) of target particle.
            if which="sum", calc dσ/dΩ(θ) of incident particle + dσ/dΩ(θ) of target particle.

    Return
    ------
    dσ/dΩ(θ) in the center-of-mass system [mb/str]
    """
    # dσ/dΩ[mb/str]
    # 10.0 : fm^2 --> mb
    if which == "inc":
        # incident particle
        if isinstance(theta, uncertainties.core.AffineScalarFunc):
            return 10.0 * (Zi * Zt * E2 / (4.0 * T))**2.0 * umath.pow(umath.sin(theta / 2.0), -4.0)
        elif isinstance(theta, np.ndarray) and isinstance(theta[0], uncertainties.core.AffineScalarFunc):
            return 10.0 * (Zi * Zt * E2 / (4.0 * T))**2.0 * unp.pow(unp.sin(theta / 2.0), -4.0)
        else:
            return 10.0 * (Zi * Zt * E2 / (4.0 * T))**2.0 * np.power(np.sin(theta / 2.0), -4.0)
    elif which == "tar":
        # target particle
        if isinstance(theta, uncertainties.core.AffineScalarFunc):
            return 10.0 * (Zi * Zt * E2 / (4.0 * T))**2.0 * umath.pow(umath.cos(theta / 2.0), -4.0)
        elif isinstance(theta, np.ndarray) and isinstance(theta[0], uncertainties.core.AffineScalarFunc):
            return 10.0 * (Zi * Zt * E2 / (4.0 * T))**2.0 * unp.pow(unp.cos(theta / 2.0), -4.0)
        else:
            return 10.0 * (Zi * Zt * E2 / (4.0 * T))**2.0 * np.power(np.cos(theta / 2.0), -4.0)
    elif which == "sum":
        # incident particle + target particle
        if isinstance(theta, uncertainties.core.AffineScalarFunc):
            return 10.0 * (Zi * Zt * E2 / (4.0 * T))**2.0 \
                * (umath.pow(umath.sin(theta / 2.0), -4.0) + umath.pow(umath.cos(theta / 2.0), -4.0))
        elif isinstance(theta, np.ndarray) and isinstance(theta[0], uncertainties.core.AffineScalarFunc):
            return 10.0 * (Zi * Zt * E2 / (4.0 * T))**2.0 \
                * (unp.pow(unp.sin(theta / 2.0), -4.0) + unp.power(unp.cos(theta / 2.0), -4.0))
        else:
            return 10.0 * (Zi * Zt * E2 / (4.0 * T))**2.0 \
                * (np.power(np.sin(theta / 2.0), -4.0) + np.power(np.cos(theta / 2.0), -4.0))
    else:
        raise ValueError(
            f"Unrecognized which option:{which}. which must be inc/tar/sum.")


def rutherford_lab(theta, T, Zi, Ai, Zt, At, which="inc"):
    """
    Rutherford Scattering in the lab system

    Arguments
    ---------
    theta : scattering angle in the lab system
    T     : kinetic energy of incident particle in the lab system
    Zi    : atomic number of incident particle; charge in units of e
    Ai    : mass number of incident particle
    Zt    : atomic number of target particle; charge in units of e
    At    : mass number of target particle
    which : if which="inc", calc dσ/dΩ(θ) of incident particle.
            if which="tar", calc dσ/dΩ(θ) of target particle.
            if which="sum", calc dσ/dΩ(θ) of incident particle + dσ/dΩ(θ) of target particle.

    Return
    ------
    dσ/dΩ(θ) in the center-of-mass system [mb/str]
    """
    # kinetic energy in the center-of-mass system
    Tcm = lab2cm.kinetic_energy(T, Ai, At)
    if which == "inc":
        # incident particle
        n = At / Ai
        # theta_lab to theta_cm
        theta_cm = lab2cm.theta(theta, n)
        # dΩcm/dΩlab
        dOcm_dOlab = cm2lab.dOmega(theta_cm, n)
        # dσ/dΩ(θ) in the lab system [mb/str]
        # dΩcm/dΩlab * dσ/dΩcm
        return dOcm_dOlab * rutherford(theta_cm, Tcm, Zi, Zt, which="inc")
    elif which == "tar":
        # target particle
        n = 1.0
        # theta_lab to theta_cm
        theta_cm = lab2cm.theta(theta, n)
        # dΩcm/dΩlab
        dOcm_dOlab = cm2lab.dOmega(theta_cm, n)
        # dσ/dΩ(θ) in the lab system [mb/str]
        # dΩcm/dΩlab * dσ/dΩcm
        return dOcm_dOlab * rutherford(theta_cm, Tcm, Zi, Zt, which="tar")
    elif which == "sum":
        # incident particle + target particle
        ni = At / Ai
        nt = 1.0
        # theta_lab to theta_cm
        theta_cmi = lab2cm.theta(theta, ni)
        theta_cmt = lab2cm.theta(theta, nt)
        # dΩcm/dΩlab
        dOcm_dOlabi = cm2lab.dOmega(theta_cmi, ni)
        dOcm_dOlabt = cm2lab.dOmega(theta_cmt, nt)
        # dσ/dΩ(θ) in the lab system [mb/str]
        # dΩcm/dΩlab * dσ/dΩcm
        dsi = dOcm_dOlabi * rutherford(theta_cmi, Tcm, Zi, Zt, which="inc")
        dst = dOcm_dOlabt * rutherford(theta_cmt, Tcm, Zi, Zt, which="tar")
        return dsi + dst
    else:
        raise ValueError(
            f"Unrecognized which option:{which}. which must be inc/tar/sum.")


def mott(theta, T, Z, A, S):
    """
    Mott Scattering in the center-of-mass system

    Arguments
    ---------
    theta : scattering angle in the center-of-mass system
    T     : kinetic energy of incident particle in the center-of-mass system
    Z     : atomic number; charge in units of e
    A     : mass number
    S     : spin

    Return
    ------
    dσ/dΩ(θ) in the center-of-mass system [mb/str]
    """
    # rest energy
    mc2 = A * AMU
    # total energy of incident/target particle in the center-of-mass system
    Ecm = (T + 2.0 * mc2) / 2.0
    # gamma (= gamma_cm in this situattion)
    gamma = Ecm / mc2
    # beta (= beta_cm in this situattion)
    beta = np.sqrt(1.0 - 1.0 / gamma**2.0)
    # relative beta in the center-of-mass system
    brel = 2.0 * beta / (1.0 + beta**2.0)

    # dσ/dΩ[mb/str]
    # 10.0 : fm^2 --> mb
    if isinstance(theta, uncertainties.core.AffineScalarFunc):
        return 10.0 * (Z**2 * E2 / (4.0 * T))**2.0 * (
            umath.pow(umath.sin(theta / 2.0), -4.0)
            + umath.pow(umath.cos(theta / 2.0), -4.0)
            + (-1.0)**(2.0 * S) * 2.0 / (2.0 * S + 1.0) *
            umath.pow(umath.sin(theta / 2.0), -2.0) *
            umath.pow(umath.cos(theta / 2.0), -2.0) *
            umath.cos(Z**2.0 * E2 / (HBARC * brel) *
                      umath.log(umath.pow(umath.tan(theta / 2.0), 2.0))
                      )
        )
    elif isinstance(theta, np.ndarray) and isinstance(theta[0], uncertainties.core.AffineScalarFunc):
        return 10.0 * (Z**2 * E2 / (4.0 * T))**2.0 * (
            unp.pow(unp.sin(theta / 2.0), -4.0)
            + unp.pow(unp.cos(theta / 2.0), -4.0)
            + (-1.0)**(2.0 * S) * 2.0 / (2.0 * S + 1.0) *
            unp.pow(unp.sin(theta / 2.0), -2.0) *
            unp.pow(unp.cos(theta / 2.0), -2.0) *
            unp.cos(Z**2.0 * E2 / (HBARC * brel) *
                    unp.log(unp.pow(unp.tan(theta / 2.0), 2.0))
                    )
        )
    else:
        return 10.0 * (Z**2 * E2 / (4.0 * T))**2.0 * (
            np.power(np.sin(theta / 2.0), -4.0)
            + np.power(np.cos(theta / 2.0), -4.0)
            + (-1.0)**(2.0 * S) * 2.0 / (2.0 * S + 1.0) *
            np.power(np.sin(theta / 2.0), -2.0) *
            np.power(np.cos(theta / 2.0), -2.0) *
            np.cos(Z**2.0 * E2 / (HBARC * brel) *
                   np.log(np.power(np.tan(theta / 2.0), 2.0))
                   )
        )


def mott_lab(theta, T, Z, A, S):
    """
    Mott Scattering in the lab system

    Arguments
    ---------
    theta : scattering angle in the lab system
    T     : kinetic energy of incident particle in the lab system
    Z     : atomic number; charge in units of e
    A     : mass number
    S     : spin

    Return
    ------
    dσ/dΩ(θ) in the lab system [mb/str]
    """
    # At/Ai = A/A = 1.0
    n = 1.0
    # theta_lab to theta_cm
    theta_cm = lab2cm.theta(theta, n)
    # kinetic energy in the center-of-mass system
    Tcm = lab2cm.kinetic_energy(T, A, A)
    # dΩcm/dΩlab
    dOcm_dOlab = cm2lab.dOmega(theta_cm, n)
    # dσ/dΩ(θ) in the lab system [mb/str]
    # dΩcm/dΩlab * dσ/dΩcm
    return dOcm_dOlab * mott(theta_cm, Tcm, Z, A, S)
