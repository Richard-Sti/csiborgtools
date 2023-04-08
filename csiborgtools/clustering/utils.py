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
import numpy


def rvs_in_sphere(nsamples, R, random_state=42, dtype=numpy.float32):
    """
    Generate random samples in a sphere of radius `R` centered at the
    origin.

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
    # Sample spherical coordinates
    r = gen.uniform(0, 1, nsamples).astype(dtype)**(1/3) * R
    theta = 2 * numpy.arcsin(gen.uniform(0, 1, nsamples).astype(dtype))
    phi = 2 * numpy.pi * gen.uniform(0, 1, nsamples).astype(dtype)
    # Convert to cartesian coordinates
    x = r * numpy.sin(theta) * numpy.cos(phi)
    y = r * numpy.sin(theta) * numpy.sin(phi)
    z = r * numpy.cos(theta)

    return numpy.vstack([x, y, z]).T