A5tools
=======

A5tools provides
- functions to find theoretical value of differential cross-section of Rutherford/Mott scattering
- utility functions to find experimental value of differential cross-section of Rutherford/Mott scattering
- functions to transform quantities between center-of-mass and laboratory systems

## Requirements

- numpy
- uncertainties

## Usage

### Theoretical value of differential cross-section
<details>
<summary>Rutherford scattering (CM)</summary>
Find theoretical value of differential cross-section of Rutherford scattering in the center-of-mass system.

Assume that
- Incident particle is C13
- Target particle is C12
- Kinetic energy is 5.0 MeV (in the center-of-mass system)

```python
import numpy as np
import matplotlib.pyplot as plt
import A5tools

T = 5.0 # Kinetic energy
Zi = 6  # Atomic number of incident particle
Zt = 6  # Atomic number of target particle
x = np.linspace(20.0, 160.0, 1000)
y = A5tools.rutherford(np.radians(x), T, Zi, Zt)
# plot
plt.plot(x, y)
plt.yscale('log')
plt.xlabel('scattering angle (CM) [deg]')
plt.ylabel('differential cross-section [mb/str]')
plt.show()
```
</details>

<details>
<summary>Rutherford scattering (Lab)</summary>
Find theoretical value of differential cross-section of Rutherford scattering in the laboratory system.

Assume that
- Incident particle is C13
- Target particle is C12
- Kinetic energy is 10.0 MeV (in the laboratory system)

```python
import numpy as np
import matplotlib.pyplot as plt
import A5tools

T = 10.0 # Kinetic energy
Zi = 6   # Atomic number of incident particle
Ai = 13  #
Zt = 6   # Atomic number of target particle
At = 12  #
x = np.linspace(10.0, 60.0, 1000)
y = A5tools.rutherford_lab(np.radians(x), T, Zi, Ai, Zt, At)
# plot
plt.plot(x, y)
plt.yscale('log')
plt.xlabel('scattering angle (Lab) [deg]')
plt.ylabel('differential cross-section [mb/str]')
plt.show()
```
</details>

<details>
<summary>Mott scattering (CM)</summary>
Find theoretical value of differential cross-section of Mott scattering in the center-of-mass system.

Assume that
- Incident/Target particle is C12
- Kinetic energy is 5.0 MeV (in the center-of-mass system)

```python
import numpy as np
import matplotlib.pyplot as plt
import A5tools

T = 5.0 # Kinetic energy
Z = 6   # Atomic number of incident/target particle
A = 12  # Mass number of incident/target particle
S = 0.0 # spin
x = np.linspace(20.0, 160.0, 1000)
y = A5tools.mott(np.radians(x), T, Z, A, S)
# plot
plt.plot(x, y)
plt.yscale('log')
plt.xlabel('scattering angle (CM) [deg]')
plt.ylabel('differential cross-section [mb/str]')
plt.show()
```
</details>

<details>
<summary>Mott scattering (Lab)</summary>
Find theoretical value of differential cross-section of Mott scattering in the laboratory system.

Assume that
- Incident/Target particle is C12
- Kinetic energy is 10.0 MeV (in the laboratory system)

```python
import numpy as np
import matplotlib.pyplot as plt
import A5tools

T = 10.0 # Kinetic energy
Z = 6   # Atomic number of incident/target particle
A = 12  # Mass number of incident/target particle
S = 0.0 # spin
x = np.linspace(10.0, 60.0, 1000)
y = A5tools.mott_lab(np.radians(x), T, Z, A, S)
# plot
plt.plot(x, y)
plt.yscale('log')
plt.xlabel('scattering angle (Lab) [deg]')
plt.ylabel('differential cross-section [mb/str]')
plt.show()
```
</details>

### Experimental value of differential cross-section
Find experimental value of differential cross-section in the center-of-mass system.

```python
import A5tools
"""
load data
data type is number, numpy.ndarray or uncertainties.uncertainties.core.AffineScalarFunc
 theta  : scattering angle in the lab system
 Ni     : number of incident particles
 Ns     : number pf scattering particles
 n      : particle number / area
 dOmega : solid angle of detector
 Ai     : mass number of incident particle
 At     : mass number of target particle
"""
# differential cross-section in the lab system
dslab = A5tools.dslab(Ni, Ns, n, dOmega)
# differential cross-section in the CM system
dscm = lab2cm.dOmega(theta, At/Ai) * dslab
# scattering angle in the CM system
heta_cm = lab2cm.theta(theta, At/Ai)
```
