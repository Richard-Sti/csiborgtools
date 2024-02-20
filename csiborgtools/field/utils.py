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
Utility functions used in the rest of the `field` module to avoid circular
imports.
"""
from numba import jit
import numpy
import healpy


def force_single_precision(x):
    """
    Attempt to convert an array `x` to float 32.
    """
    if x.dtype != numpy.float32:
        x = x.astype(numpy.float32)
    return x


@jit(nopython=True)
def divide_nonzero(field0, field1):
    """
    Perform in-place `field0 /= field1` but only where `field1 != 0`.
    """
    assert field0.shape == field1.shape, "Field shapes must match."

    imax, jmax, kmax = field0.shape
    for i in range(imax):
        for j in range(jmax):
            for k in range(kmax):
                if field1[i, j, k] != 0:
                    field0[i, j, k] /= field1[i, j, k]


def nside2radec(nside):
    """
    Generate RA [0, 360] deg. and declination [-90, 90] deg for HEALPix pixel
    centres at a given nside.

    Parameters
    ----------
    nside : int
        HEALPix nside.

    Returns
    -------
    angpos : 2-dimensional array of shape (npix, 2)
    """
    pixs = numpy.arange(healpy.nside2npix(nside))
    theta, phi = healpy.pix2ang(nside, pixs)

    ra = 180 / numpy.pi * phi
    dec = 90 - 180 / numpy.pi * theta

    return numpy.vstack([ra, dec]).T
