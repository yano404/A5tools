"""
functions to correct
* θ   : scattering angle
* D    : distance between target and detector
* ΔS  : detection area
* ΔΩ : solid angle of detector
"""

import numpy as np
import uncertainties
from uncertainties import umath
import uncertainties.unumpy as unp


class correct:
    def __init__(self, R, r, AT, d, w, h):
        """
        R     : radius of rotation [mm]
        r     : distance between target and the center of rotation [mm]
        AT    : α_t the angle between α=0 and target [rad]
        d,w,h : depth, width, height of slit immplemented on detector [mm]
        """
        self.R = R
        self.r = r
        self.AT = AT
        self.d = d
        self.w = w
        self.h = h
        if sum([isinstance(x, uncertainties.core.Variable) for x in [R, r, AT, d, w, h]]) > 0:
            self.uncertain = True
        else:
            self.uncertain = False

    def D(self, a):
        """
        Arguments
        ---------
        a : angle of rotation [rad]

        Return
        ------
        D : distance between target and detector [mm]
        """
        Rd = self.R + self.d
        r = self.r
        AT = self.AT
        if isinstance(a, (int, float, np.number, uncertainties.core.AffineScalarFunc)):
            return umath.sqrt(Rd**2.0 + r**2.0 - 2.0 * Rd * r * umath.cos(a + AT))
        elif isinstance(a, np.ndarray):
            return unp.sqrt(Rd**2.0 + r**2.0 - 2.0 * Rd * r * unp.cos(a + AT))
        else:
            raise ValueError(
                f"This function not supported for the input types. argument 'a' must be number or array")

    def theta(self, a):
        """
        Arguments
        ---------
        a : angle of rotation [rad]

        Return
        ------
        θ : scattering angle in the lab system [rad]
        """
        D0 = self.D(0.0)
        D = self.D(a)
        Rd = self.R + self.d
        if isinstance(a, (int, float, np.number, uncertainties.core.AffineScalarFunc)):
            return np.sign(a) * umath.acos(
                (D0**2.0 + D**2.0 - 2.0 * Rd**2.0 *
                 (1.0 - umath.cos(a))) / (2.0 * D0 * D)
            )
        elif isinstance(a, np.ndarray):
            return np.sign(a) * unp.arccos(
                (D0**2.0 + D**2.0 - 2.0 * Rd**2.0 *
                 (1.0 - unp.cos(a))) / (2.0 * D0 * D)
            )
        else:
            raise ValueError(
                f"This function not supported for the input types. argument 'a' must be number or array")

    def dS(self, a):
        """
        Arguments
        ---------
        a : angle of rotation [rad]

        Return
        ------
        ΔS : detection area [mm^2]
        """
        Rd = self.R + self.d
        r = self.r
        AT = self.AT
        D0 = self.D(0.0)
        A0 = umath.acos((Rd**2.0 + D0**2.0 - r**2.0) / (2.0 * Rd * D0))
        d = self.d
        w = self.w
        h = self.h
        if isinstance(a, (int, float, np.number, uncertainties.core.AffineScalarFunc)):
            phi = umath.fabs(a - A0 - self.theta(a))
            # ΔS
            return w * h * umath.cos(phi) - d * h * umath.sin(phi)
        elif isinstance(a, np.ndarray):
            phi = unp.fabs(a - A0 - self.theta(a))
            return w * h * unp.cos(phi) - d * h * unp.sin(phi)
        else:
            raise ValueError(
                f"This function not supported for the input types. argument 'a' must be number or array")

    def dOmega(self, a):
        """
        Arguments
        ---------
        a : angle of rotation [rad]

        Return
        ------
        ΔΩ : solid angle of detector [str]
        """
        return self.dS(a) / self.D(a)**2.0
