"""
utils.py
Utilities for analysis.
Trey V. Wenger
"""

import numpy as np

# Reid+2019 A5 rotation model parameters
__R0 = 8.166  # kpc
__Usun = 10.449  # km/s
__Vsun = 12.092  # km/s
__Wsun = 7.729  # km/s
__a2 = 0.977
__a3 = 1.623

# IAU defined LSR
__Ustd = 10.27  # km/s
__Vstd = 15.32  # km/s
__Wstd = 7.74  # km/s


def reid19_theta(R, R0=__R0, a2=__a2, a3=__a3):
    """
    Calculate the Reid et al. (2019) circular rotation speed at a
    given Galactocentric radius.

    Inputs:
        R :: scalar (kpc)
            Galactocentric radius
        R0 :: scalar (kpc)
            Solar Galactocentric radius
        a2, a3 :: scalar
            Parameters that define rotation curve

    Returns: theta
        theta :: scalar (km/s)
            Circular rotation speed
    """
    rho = R / (a2 * R0)
    lam = (a3 / 1.5) ** 5.0
    loglam = np.log10(lam)
    term1 = 200.0 * lam**0.41
    term2 = np.sqrt(
        0.8 + 0.49 * loglam + 0.75 * np.exp(-0.4 * lam) / (0.47 + 2.25 * lam**0.4)
    )
    term3 = (0.72 + 0.44 * loglam) * 1.97 * rho**1.22 / (rho**2.0 + 0.61) ** 1.43
    term4 = 1.6 * np.exp(-0.4 * lam) * rho**2.0 / (rho**2.0 + 2.25 * lam**0.4)
    theta = term1 / term2 * np.sqrt(term3 + term4)
    return theta


__theta0 = reid19_theta(__R0, R0=__R0, a2=__a2, a3=__a3)


def reid19_vlsr(
    glong,
    glat,
    R,
    R0=__R0,
    a2=__a2,
    a3=__a3,
    Usun=__Usun,
    Vsun=__Vsun,
    Wsun=__Wsun,
    theta0=__theta0,
):
    """
    Calculate the Reid et al. (2019) rotation curve LSR velocity
    at a given position.

    Inputs:
        glong, glat :: scalars (deg)
            Galactic longitude and latitude
        R :: scalar (kpc)
            Galactocentric radius
        R0 :: scalar (kpc)
            Solar Galactocentric radius
        a2, a3 :: scalar
            Parameters that define rotation curve
        Usun, Vsun, Wsun :: scalars (km/s)
            Solar motion relative to the LSR
        theta0 :: scalar (km/s)
            Solar orbital speed

    Returns: vlsr
        vlsr :: scalar (km/s)
            LSR velocity
    """
    # Circular velocities
    theta = reid19_theta(R, R0=R0, a2=a2, a3=a3)

    # Radial velocity relative to LSR
    sin_glong = np.sin(glong)
    cos_glat = np.cos(glat)
    vlsr = R0 * sin_glong * cos_glat * (theta / R - theta0 / R0)

    # Difference between solar motion and IAU definition
    U = (__Ustd - Usun) * np.cos(np.deg2rad(glong))
    V = (__Vstd - Vsun) * sin_glong
    W = (__Wstd - Wsun) * np.sin(np.deg2rad(glat))
    return vlsr + (U + V) * cos_glat + W
