"""
functions to correct
* θ   : scattering angle
* D    : distance between target and detector
* ΔS  : detection area
* ΔΩ : solid angle of detector
"""

import numpy as np
import uncertainties
import uncertainties.unumpy as unp
import uncertainties.unumpy as umath


class correct:
    def __init__(self, R, r, AT, d, w, h):
        """
        R     : radius of rotation
        r     : distance between target and the center of rotation
        AT    : α_t the angle between α=0 and target
        d,w,h : depth, width, height of slit immplemented on detector
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
        Rd = self.R + self.d
        r = self.r
        AT = self.AT
        if isinstance(a, uncertainties.core.Variable):
            return umath.sqrt(Rd**2.0 + r**2.0 - 2.0 * Rd * r * umath.cos(a + AT))
        else:
            return unp.sqrt(Rd**2.0 + r**2.0 - 2.0 * Rd * r * unp.cos(a + AT))

    def theta(self, a):
        D0 = self.D(0.0)
        D = self.D(a)
        Rd = self.R + self.d
        if isinstance(a, uncertainties.core.Variable):
            return np.sign(a) * umath.acos(
                (D0**2.0 + D**2.0 - 2.0 * Rd**2.0 *
                 (1.0 - umath.cos(a))) / (2.0 * D0 * D)
            )
        else:
            return np.sign(a) * unp.arccos(
                (D0**2.0 + D**2.0 - 2.0 * Rd**2.0 *
                 (1.0 - unp.cos(a))) / (2.0 * D0 * D)
            )

    def dS(self, a):
        Rd = self.R + self.d
        r = self.r
        AT = self.AT
        D0 = self.D(0.0)
        A0 = umath.acos((Rd**2.0 + D0**2.0 - r**2.0) / (2.0 * Rd * D0))
        d = self.d
        w = self.w
        h = self.h
        if isinstance(a, uncertainties.core.Variable):
            phi = umath.fabs(a - A0 - self.theta(a))
            # ΔS
            return w * h * umath.cos(phi) - d * h * umath.sin(phi)
        else:
            phi = unp.fabs(a - A0 - self.theta(a))
            return w * h * unp.cos(phi) - d * h * unp.sin(phi)

    def dOmega(self, a):
        return self.dS(a) / self.D(a)**2.0
