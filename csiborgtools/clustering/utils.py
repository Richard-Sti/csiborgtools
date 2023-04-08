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
"""Clustering support functions."""
from warnings import warn
import numpy


###############################################################################
#                            Random points                                    #
###############################################################################


def rvs_in_sphere(nsamples, R, random_state=42, dtype=numpy.float32):
    """
    Generate random uniform samples in a sphere of radius `R` in Cartesian
    coordinates.

    Parameters
    ----------
    nsamples : int
        Number of samples to generate.
    R : float
        Radius of the sphere.
    random_state : int, optional
        Random state for the random number generator.
    dtype : numpy dtype, optional
        Data type, by default `numpy.float32`.

    Returns
    -------
    samples : 2-dimensional array of shape `(nsamples, 3)`
    """
    gen = numpy.random.default_rng(random_state)
    # Spherical
    r = gen.random(nsamples, dtype=dtype)**(1/3) * R
    theta = 2 * numpy.arcsin(gen.random(nsamples, dtype=dtype))
    phi = 2 * numpy.pi * gen.random(nsamples, dtype=dtype)
    # Cartesian
    x = r * numpy.sin(theta) * numpy.cos(phi)
    y = r * numpy.sin(theta) * numpy.sin(phi)
    z = r * numpy.cos(theta)

    return numpy.vstack([x, y, z]).T


def rvs_on_sphere(nsamples, indeg, random_state=42, dtype=numpy.float32):
    r"""
    Generate random uniform samples on the surface of a sphere.

    Parameters
    ----------
    nsamples : int
        Number of samples to generate.
    indeg : bool, optional
        Whether to return the right ascension and declination in degrees.
    random_state : int
        Random state for the random number generator.
    dtype : numpy dtype, optional
        Data type, by default `numpy.float32`.

    Returns
    -------
    out : 2-dimensional array of shape `(nsamples, 2)`
        RA in :math:`[0, 2\pi)` and dec in :math:`[-\pi / 2, \pi / 2]`,
        respectively. If `indeg` then converted to degrees.
    """
    gen = numpy.random.default_rng(random_state)
    ra = 2 * numpy.pi * gen.random(nsamples, dtype=dtype)
    dec = numpy.arcsin(2 * (gen.random(nsamples, dtype=dtype) - 0.5))
    if indeg:
        ra = numpy.deg2rad(ra)
        dec = numpy.deg2rad(dec)
    return numpy.vstack([ra, dec]).T

###############################################################################
#                               RA wrapping                                   #
###############################################################################

def wrapRA(ra, indeg):
    """
    Wrap RA from :math:`[-180, 180)` to :math`[0, 360)` degrees if `indeg` or
    equivalently in radians otherwise.

    Paramaters
    ----------
    ra : 1-dimensional array
        Right ascension.
    indeg : bool
        Whether the right ascension is in degrees.

    Returns
    -------
    wrapped_ra : 1-dimensional array
    """
    mask = ra < 0
    if numpy.sum(mask) == 0:
        warn("No negative right ascension found.", UserWarning())
    ra[mask] += 360 if indeg else 2 * numpy.pi
    return ra