# Copyright (C) 2022 Richard Stiskalek, Deaglan Bartlett
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
"""
What is life about?
"""

import numpy
# from ..io import open_particle


def nfw_profile(r, Rs, rho0):
    r"""
    The Navarro-Frenk-White (NFW) profile defined as

    .. math::
        \rho(r) = \frac{\rho_0}{x(1 + x)^2}

    where :math:`x = r / R_s` with free parameters :math:`R_s, \rho_0`.

    Parameters
    ----------
    r : float
        Radial distance :math:`r`.
    Rs : float
        Scale radius :math:`R_s`.
    rho0 : float
        NFW density parameter :math:`\rho_0`.

    Returns
    -------
    density : float
        Density of the NFW profile at :math:`r`.
    """
    x = r / Rs
    return rho0 / (x * (1 + x)**2)


def nfw_mass(r, Rs, rho0):
    r"""
    Enclosed mass  of a NFW profile in radius :math:`r`.

    Parameters
    ----------
    r : float
        Radial distance :math:`r`.
    Rs : float
        Scale radius :math:`R_s`.
    rho0 : float
        NFW density parameter :math:`\rho_0`.

    Returns
    -------
    M : float
        The enclosed mass.
    """
    x = r / Rs
    return 4 * numpy.pi * Rs**3 * rho0 * (numpy.log(1 + x) - r / (r + Rs))
