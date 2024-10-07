# Copyright (C) 2024 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Various cosmography functions for converting between distance indicators."""
from jax import numpy as jnp

from ..params import SPEED_OF_LIGHT

H0 = 100  # km / s / Mpc


def dist2redshift(dist, Omega_m, h=1.):
    """
    Convert comoving distance to cosmological redshift if the Universe is
    flat and z << 1.
    """
    eta = 3 * Omega_m / 2
    return 1 / eta * (1 - (1 - 2 * 100 * h * dist / SPEED_OF_LIGHT * eta)**0.5)


def redshift2dist(z, Omega_m):
    """
    Convert cosmological redshift to comoving distance if the Universe is
    flat and z << 1.
    """
    q0 = 3 * Omega_m / 2 - 1
    return SPEED_OF_LIGHT * z / (2 * H0) * (2 - z * (1 + q0))


def gradient_redshift2dist(z, Omega_m):
    """
    Gradient of the redshift to comoving distance conversion if the Universe is
    flat and z << 1.
    """
    q0 = 3 * Omega_m / 2 - 1
    return SPEED_OF_LIGHT / H0 * (1 - z * (1 + q0))


def distmod2dist(mu, Om0):
    """
    Convert distance modulus to distance in `Mpc / h`. The expression is valid
    for a flat universe over the range of 0.00001 < z < 0.1.
    """
    term1 = jnp.exp((0.443288 * mu) + (-14.286531))
    term2 = (0.506973 * mu) + 12.954633
    term3 = ((0.028134 * mu) ** (
        ((0.684713 * mu)
         + ((0.151020 * mu) + (1.235158 * Om0))) - jnp.exp(0.072229 * mu)))
    term4 = (-0.045160) * mu
    return (-0.000301) + (term1 * (term2 - (term3 - term4)))


def distmod2dist_gradient(mu, Om0):
    """
    Calculate the derivative of comoving distance in `Mpc / h` with respect to
    the distance modulus. The expression is valid for a flat universe over the
    range of 0.00001 < z < 0.1.
    """
    term1 = jnp.exp((0.443288 * mu) + (-14.286531))
    dterm1 = 0.443288 * term1

    term2 = (0.506973 * mu) + 12.954633
    dterm2 = 0.506973

    term3 = ((0.028134 * mu)**(((0.684713 * mu) + ((0.151020 * mu) + (1.235158 * Om0))) - jnp.exp(0.072229 * mu)))  # noqa
    ln_base = jnp.log(0.028134) + jnp.log(mu)
    exponent = 0.835733 * mu + 1.235158 * Om0 - jnp.exp(0.072229 * mu)
    exponent_derivative = 0.835733 - 0.072229 * jnp.exp(0.072229 * mu)
    dterm3 = term3 * ((1 / mu) * exponent + exponent_derivative * ln_base)

    term4 = (-0.045160) * mu
    dterm4 = -0.045160

    return (dterm1 * (term2 - (term3 - term4))
            + term1 * (dterm2 - (dterm3 - dterm4)))


def distmod2redshift(mu, Om0):
    """
    Convert distance modulus to redshift, assuming `h = 1`. The expression is
    valid for a flat universe over the range of 0.00001 < z < 0.1.
    """
    return jnp.exp(((0.461108 * mu) - ((0.022187 * Om0) + (((0.022347 * mu)** (12.631788 - ((-6.708757) * Om0))) + 19.529852))))  # noqa
