"""
functions to find observed value of differential cross-section
"""


from .constants import *


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
    q: value of current digityzer in units of 1.0E-10[C]
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
    return rhodx / At * NA
