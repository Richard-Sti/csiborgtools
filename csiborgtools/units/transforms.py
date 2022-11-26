# Copyright (C) 2022 Richard Stiskalek
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
Various coordinate transformations.
"""

import numpy


def cartesian_to_radec(x, y, z):
    """
    Calculate the radial distance, right ascension in [0, 360) degrees and
    declination [-90, 90] degrees. Note, the observer should be placed in the
    middle of the box.

    Parameters
    ----------
    x, y, z : 1-dimensional arrays
        Cartesian coordinates.
    Returns
    -------
    dist, ra, dec : 1-dimensional arrays
        Radial distance, right ascension and declination.
    """
    dist = numpy.sqrt(x**2 + y**2 + z**2)
    dec = numpy.rad2deg(numpy.arcsin(z/dist))
    ra = numpy.rad2deg(numpy.arctan2(y, x))
    # Make sure RA in the correct range
    ra[ra < 0] += 360
    return dist, ra, dec


def radec_to_cartesian(dist, ra, dec, isdeg=True):
    """
    Convert distance, right ascension and declination to Cartesian coordinates.

    Parameters
    ----------
    dist, ra, dec : 1-dimensional arrays
        The spherical coordinates.
    isdeg : bool, optional
        Whether `ra` and `dec` are in degres. By default `True`.

    Returns
    -------
    x, y, z : 1-dimensional arrays
        Cartesian coordinates.
    """
    if isdeg:
        ra = numpy.deg2rad(ra)
        dec = numpy.deg2rad(dec)
    x = dist * numpy.cos(dec) * numpy.cos(ra)
    y = dist * numpy.cos(dec) * numpy.sin(ra)
    z = dist * numpy.sin(dec)
    return x, y, z
