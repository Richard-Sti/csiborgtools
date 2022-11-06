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


def cartesian_to_radec(arr, xpar="peak_x", ypar="peak_y", zpar="peak_z"):
    r"""
    Extract `x`, `y`, and `z` coordinates from a record array `arr` and
    calculate the radial distance :math:`r` in coordinate units, right
    ascension :math:`\mathrm{RA} \in [0, 360)` degrees and declination
    :math:`\delta \in [-90, 90]` degrees.

    Parameters
    ----------
    arr : record array
        Record array with the Cartesian coordinates.
    xpar : str, optional
        Name of the x coordinate in the record array.
    ypar : str, optional
        Name of the y coordinate in the record array.
    zpar : str, optional
        Name of the z coordinate in the record array.

    Returns
    -------
    dist : 1-dimensional array
        Radial distance.
    ra : 1-dimensional array
        Right ascension.
    dec : 1-dimensional array
        Declination.
    """
    x, y, z = arr[xpar], arr[ypar], arr[zpar]

    dist = numpy.sqrt(x**2 + y**2 + z**2)
    dec = numpy.rad2deg(numpy.arcsin(z/dist))
    ra = numpy.rad2deg(numpy.arctan2(y, x))
    # Make sure RA in the correct range
    ra[ra < 0] += 360

    return dist, ra, dec
