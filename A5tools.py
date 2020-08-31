"""
A5tools
=======
Author: Takayuki YANO
"""

import numpy as np

"""
Constants
"""
alpha = 7.2973525693E-3  # fine-structure constant
hbarc = 197.3269804  # ħc [Mev fm]
e2 = hbarc * alpha  # e^2 [Mev fm]
amu = 931.49410242  # atomic mass unit [MeV/c^2]
Na = 6.02214076E+23  # Avogadro constant [mol^(-1)]

"""
Functions to find Theoretical value of differential cross-section
"""
# Theoretical value of Differential cross-section in the center-of-mass system


def rutherford(theta, Tlab, Zi, Ai, Zt, At):
    """
    Rutherford Scattering

    Arguments
    ---------
    theta : scattering angle in the center-of-mass system
    Tlab  : kinetic energy of incident particle in the lab system
    Zi    : atomic number of incident particle; charge in units of e
    Ai    : mass number of incident particle
    Zt    : atomic number of target particle; charge in units of e
    At    : mass number of target particle

    Return
    ------
    dσ/dΩ(θ) in the center-of-mass system [mb/str]
    """

    # mc^2
    mic2 = Ai * amu
    mtc2 = At * amu

    # energy of incident particle in the lab system
    Eilab = Tlab + mic2

    # calc T in the center-of-mass system
    Ecm = np.sqrt(2.0 * Eilab * mtc2 + mic2**2.0 + mtc2**2.0)
    Tcm = Ecm - mic2 - mtc2

    # calc dσ/dΩ[mb/str]
    # 10.0 : fm^2 --> mb
    return 10.0 * (Zi * Zt * e2 / (4.0 * Tcm))**2.0 * np.power(np.sin(theta / 2.0), -4.0)


def mott(theta, Tlab, Z, A, S):
    """
    Mott Scattering

    Arguments
    ---------
    theta : scattering angle in the center-of-mass system
    Tlab  : kinetic energy of incident particle in the lab system
    Z     : atomic number; charge in units of e
    A     : mass number
    S     : spin

    Return
    ------
    dσ/dΩ(θ) in the center-of-mass system [mb/str]
    """

    # mc^2
    mc2 = A * amu

    # energy of incident particle in the lab system
    Eilab = Tlab + mc2

    # calc T in the center-of-mass system
    Ecm = np.sqrt(2.0 * Eilab * mc2 + 2.0 * mc2**2.0)
    Tcm = Ecm - 2.0 * mc2

    # calc relative beta
    brel = np.sqrt((Tlab + mc2)**2.0 - mc2**2.0) / (Tlab + mc2)

    # calc dσ/dΩ[mb/str]
    # 10.0 : fm^2 --> mb
    return 10.0 * (Z**2 * e2 / (4.0 * Tcm))**2.0 * (
        np.power(np.sin(theta / 2.0), -4.0)
        + np.power(np.cos(theta / 2.0), -4.0)
        + (-1.0)**(2.0 * S) * 2.0 / (2.0 * S + 1.0) *
        np.power(np.sin(theta / 2.0), -2.0) *
        np.power(np.cos(theta / 2.0), -2.0) *
        np.cos(Z**2.0 * e2 / (hbarc * brel) *
               np.log(np.power(np.tan(theta / 2.0), 2.0))
               )
    )

# Theoretical value of Differential cross-section in the lab system


def rutherford_lab(theta_lab, Tlab, Zi, Ai, Zt, At):
    """
    Rutherford Scattering in the lab system

    Arguments
    ---------
    theta_lab : scattering angle in the lab system
    Tlab  : kinetic energy of incident particle in the lab system
    Zi    : atomic number of incident particle; charge in units of e
    Ai    : mass number of incident particle
    Zt    : atomic number of target particle; charge in units of e
    At    : mass number of target particle

    Return
    ------
    dσ/dΩ(θ) in the lab system [mb/str]
    """
    n = At / Ai
    theta_cm, dOlab_dOcm = lab2cm(theta_lab, n)
    return rutherford(theta_cm, Tlab, Zi, Ai, Zt, At) * np.power(dOlab_dOcm, -1.0)


def rutherford_lab_target(theta_lab, Tlab, Zi, Ai, Zt, At):
    """
    Rutherford Scattering in the lab system

    Arguments
    ---------
    theta_lab : scattering angle in the lab system
    Tlab  : kinetic energy of incident particle in the lab system
    Zi    : atomic number of incident particle; charge in units of e
    Ai    : mass number of incident particle
    Zt    : atomic number of target particle; charge in units of e
    At    : mass number of target particle

    Return
    ------
    differential cross-section of target particle in the lab system
    dσi/dΩ(θ) + dσt/dΩ(θ) in the lab system [mb/str]
    """
    theta_cmt, dOlab_dOcmt = lab2cm(theta_lab, 1.0)
    dst = rutherford(np.pi - theta_cmt, Tlab, Zi, Ai, Zt,
                     At) * np.power(dOlab_dOcmt, -1.0)
    return dst


def rutherford_lab_sum(theta_lab, Tlab, Zi, Ai, Zt, At):
    """
    Rutherford Scattering

    Arguments
    ---------
    theta_lab : scattering angle in the lab system
    Tlab  : kinetic energy of incident particle in the lab system
    Zi    : atomic number of incident particle; charge in units of e
    Ai    : mass number of incident particle
    Zt    : atomic number of target particle; charge in units of e
    At    : mass number of target particle

    Return
    ------
    Sum of differential cross-section
    dσi/dΩ(θ) + dσt/dΩ(θ) in the lab system [mb/str]
    """
    n = At / Ai
    theta_cmi, dOlab_dOcmi = lab2cm(theta_lab, n)
    theta_cmt, dOlab_dOcmt = lab2cm(theta_lab, 1.0)
    dsi = rutherford(theta_cmi, Tlab, Zi, Ai, Zt, At) * \
        np.power(dOlab_dOcmi, -1.0)
    dst = rutherford(np.pi - theta_cmt, Tlab, Zi, Ai, Zt,
                     At) * np.power(dOlab_dOcmt, -1.0)
    return dsi + dst


def mott_lab(theta_lab, Tlab, Z, A, S):
    """
    Mott Scattering

    Arguments
    ---------
    theta_lab : scattering angle in the lab system
    Tlab  : kinetic energy of incident particle in the lab system
    Z     : atomic number; charge in units of e
    A     : mass number
    S     : spin

    Return
    ------
    dσ/dΩ(θ) in the lab system [mb/str]
    """
    n = 1.0
    theta_cm, dOlab_dOcm = lab2cm(theta_lab, n)
    return mott(theta_cm, Tlab, Z, A, S) * np.power(dOlab_dOcm, -1.0)


"""
Functions to convert
    center-of-mass system to lab system
or
    lab system to center-of-mass system
"""


def cm2lab(theta_cm, n):
    """
    function to convert center-of-mass system to lab system

    Arguments
    ---------
    theta_cm : scattering angle in the center-of-mass system [rad]
    n        : A_t / A_i; A_t, A_i means mass number of target particle or incident particle

    Return
    ------
    theta_lab : scattering angle in the lab system [rad]
      dΩcm
     -------- : factor to convert differential cross-section in the center-of-mass system to differential cross-section in the lab system
      dΩlab

    Notice
    ------
    This function do not consider relativity
    """
    theta_lab = np.arctan(np.sin(theta_cm) / (1 / n + np.cos(theta_cm)))
    dOcm_dOlab = np.power(1 + 2 * np.cos(theta_cm) / n +
                          1 / n**2, 3 / 2) / (1 + np.cos(theta_cm) / n)
    return np.array([theta_lab, dOcm_dOlab])

# function to convert lab system to center-of-mass system


def lab2cm(theta_lab, n):
    """
    function to convert lab system to center-of-mass system

    Arguments
    ---------
    theta_lab : scattering angle in the lab system [rad]
    n         : A_t / A_i; A_t, A_i means mass number of target particle or incident particle

    Return
    ------
    theta_cm  : scattering angle in the center-of-mass system [rad]
      dΩlab
     -------- : factor to convert differential cross-section in the lab system to differential cross-section in the center-of-mass system
      dΩcm

    Notice
    ------
    This function do not consider relativity
    """
    coslab2 = np.power(np.cos(theta_lab), 2.0)
    coscm = (coslab2 - 1.0) / n + np.sqrt((1.0 - 1.0 / n**2.0)
                                          * coslab2 + np.power(coslab2 / n, 2.0))
    theta_cm = np.sign(theta_lab) * np.arccos(coscm)
    dOlab_dOcm = np.power(
        2.0 * np.cos(theta_lab) / n
        + (1.0 + np.cos(2.0 * theta_lab) / (n**2.0)) /
        np.sqrt(1.0 - np.power(np.sin(theta_lab) / n, 2.0)),
        -1.0
    )
    return np.array([theta_cm, dOlab_dOcm])


"""
Functions to analysis experimental data
"""


def gauss_int(con, sigma):
    """
    Gaussian Integral to find N_scat
    sigma must be greater than 0
    """
    return np.sqrt(2.0 * np.pi) * con * sigma


def dslab(Ni, Ns, n, dOmega, cor_factor=1):
    """
    function to find dσ/dΩ[mb/str] in the lab system

    Arguments
    ---------
    Ni : number of incident particle
    Ns : number of scattering particle
    n : particle number / area [/mb]
    dOmega : solid angle [str]

    Return
    ------
    dσ/dΩ[mb/str] in the lab system
    """
    return Ns / (Ni * n * dOmega) * cor_factor


def N_inc(q, n):
    """
    function to find Number of incident particle

    Arguments
    ---------
    q: value of current digityzer in units of e
    n: charge of incident particle in units of e

    Return
    ------
    Ni : Number of incident particle
    """
    e = 1.602176634E-19  # elementary charge [C]
    return q * 1.0E-10 / (n * e)


def narea(rhodx, At):
    """
    function to find number areal density

    Arguments
    ---------
    rhodx : mass areal density [g/mb]
    At: mass number of target particle

    Return
    ------
    number areal density [/mb]
    """
    return rhodx / At * Na


def bethe(z, a, Z, A, E_a):
    """
    function to calc -dE/d(ρx) [MeV mb / g]

    Arguments
    ---------
    z: charge of incident particle in units of e [Int]
    a: Atomic Weight of incident particle [Int]
    Z: atomic number of absorbing material [Int]
    A: Atomic Weight of absorbing material [Int]
    E_a: Energy of incident particle [MeV]

    Return
    ------
    -dE/d(ρx) [MeV mb / g]
    """
    mec2 = 0.5109990  # [MeV] m_e c^2 = 0.5109990 [MeV]
    # Constant part
    # const = 0.1535E-3 # [MeV cm^2 / mg]
    # const = 0.1535 # [MeV cm^2 / g]
    const = 0.1535E+27  # [MeV mb / g]
    # calc gamma
    gamma = (E_a / amu + a) / a
    # calc beta
    beta = np.sqrt(1.0 - 1.0 / gamma**2.0)
    # eta
    eta = beta * gamma
    # wmax
    s = mec2 / a / amu
    wmax = (2.0 * mec2 * eta**2.0) / \
        (1.0 + 2.0 * s * np.sqrt(1 + eta**2.0) + s**2.0)
    # I
    if Z < 13:
        I = (12 * Z + 7) * 1E-6  # [MeV]
    else:
        I = (9.76 * Z + 58.8 * np.pow(Z, -0.19)) * 1E-6  # [MeV]
    # return -dE/d(ρx)
    return - const * (Z / A) * (z**2.0 / beta**2.0) * (np.log(2 * mec2 * gamma**2.0 * beta**2.0 * wmax / I**2.0) - 2.0 * beta**2.0)
