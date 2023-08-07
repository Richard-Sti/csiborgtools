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
"""Collection of stand-off utility functions used in the scripts."""
import numpy
from numba import jit


@jit(nopython=True, fastmath=True, boundscheck=False)
def center_of_mass(points, mass, boxsize):
    """
    Calculate the center of mass of a halo while assuming periodic boundary
    conditions of a cubical box. Assuming that particle positions are in
    `[0, boxsize)` range. This is a JIT implementation.

    Parameters
    ----------
    points : 2-dimensional array of shape (n_particles, 3)
        Particle position array.
    mass : 1-dimensional array of shape `(n_particles, )`
        Particle mass array.
    boxsize : float
        Box size in the same units as `parts` coordinates.

    Returns
    -------
    cm : 1-dimensional array of shape `(3, )`
    """
    cm = numpy.zeros(3, dtype=points.dtype)
    totmass = sum(mass)

    # Convert positions to unit circle coordinates in the complex plane,
    # calculate the weighted average and convert it back to box coordinates.
    for i in range(3):
        cm_i = sum(mass * numpy.exp(2j * numpy.pi * points[:, i] / boxsize))
        cm_i /= totmass

        cm_i = numpy.arctan2(cm_i.imag, cm_i.real) * boxsize / (2 * numpy.pi)

        if cm_i < 0:
            cm_i += boxsize
        cm[i] = cm_i

    return cm


@jit(nopython=True)
def periodic_distance(points, reference, boxsize):
    """
    Compute the 3D distance between multiple points and a reference point using
    periodic boundary conditions. This is an optimized JIT implementation.

    Parameters
    ----------
    points : 2-dimensional array of shape `(n_points, 3)`
        Points to calculate the distance from the reference point.
    reference : 1-dimensional array of shape `(3, )`
        Reference point.
    boxsize : float
        Box size.

    Returns
    -------
    dist : 1-dimensional array of shape `(n_points, )`
    """
    npoints = len(points)
    half_box = boxsize / 2

    dist = numpy.zeros(npoints, dtype=points.dtype)
    for i in range(npoints):
        for j in range(3):
            dist_1d = abs(points[i, j] - reference[j])

            if dist_1d > (half_box):
                dist_1d = boxsize - dist_1d

            dist[i] += dist_1d**2

        dist[i] = dist[i]**0.5

    return dist


@jit(nopython=True, fastmath=True, boundscheck=False)
def delta2ncells(delta):
    """
    Calculate the number of cells in `delta` that are non-zero.

    Parameters
    ----------
    delta : 3-dimensional array
        Halo density field.

    Returns
    -------
    ncells : int
        Number of non-zero cells.
    """
    tot = 0
    imax, jmax, kmax = delta.shape
    for i in range(imax):
        for j in range(jmax):
            for k in range(kmax):
                if delta[i, j, k] > 0:
                    tot += 1
    return tot


@jit(nopython=True, fastmath=True, boundscheck=False)
def number_counts(x, bin_edges):
    """
    Calculate counts of samples in bins.

    Parameters
    ----------
    x : 1-dimensional array
        Samples' values.
    bin_edges : 1-dimensional array
        Bin edges.

    Returns
    -------
    counts : 1-dimensional array
        Bin counts.
    """
    out = numpy.full(bin_edges.size - 1, numpy.nan, dtype=numpy.float32)
    for i in range(bin_edges.size - 1):
        out[i] = numpy.sum((x >= bin_edges[i]) & (x < bin_edges[i + 1]))
    return out
