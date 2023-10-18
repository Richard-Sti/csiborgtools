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
from datetime import datetime

###############################################################################
#                           Positions                                         #
###############################################################################


@jit(nopython=True, fastmath=True, boundscheck=False)
def center_of_mass(particle_positions, particles_mass, boxsize):
    """
    Calculate the center of mass of a halo while assuming periodic boundary
    conditions of a cubical box.
    """
    cm = numpy.zeros(3, dtype=particle_positions.dtype)
    totmass = sum(particles_mass)

    # Convert positions to unit circle coordinates in the complex plane,
    # calculate the weighted average and convert it back to box coordinates.
    for i in range(3):
        cm_i = sum(particles_mass * numpy.exp(
            2j * numpy.pi * particle_positions[:, i] / boxsize))
        cm_i /= totmass

        cm_i = numpy.arctan2(cm_i.imag, cm_i.real) * boxsize / (2 * numpy.pi)

        if cm_i < 0:
            cm_i += boxsize
        cm[i] = cm_i

    return cm


@jit(nopython=True, fastmath=True, boundscheck=False)
def periodic_distance(points, reference_point, boxsize):
    """
    Compute the 3D distance between multiple points and a reference point using
    periodic boundary conditions.
    """
    npoints = len(points)
    half_box = boxsize / 2

    dist = numpy.zeros(npoints, dtype=points.dtype)
    for i in range(npoints):
        for j in range(3):
            dist_1d = abs(points[i, j] - reference_point[j])

            if dist_1d > (half_box):
                dist_1d = boxsize - dist_1d

            dist[i] += dist_1d**2

        dist[i] = dist[i]**0.5

    return dist


@jit(nopython=True, fastmath=True, boundscheck=False)
def periodic_distance_two_points(p1, p2, boxsize):
    """Compute the 3D distance between two points in a periodic box."""
    half_box = boxsize / 2

    dist = 0
    for i in range(3):
        dist_1d = abs(p1[i] - p2[i])

        if dist_1d > (half_box):
            dist_1d = boxsize - dist_1d

        dist += dist_1d**2

    return dist**0.5


@jit(nopython=True, boundscheck=False)
def periodic_wrap_grid(pos, boxsize=1):
    """Wrap positions in a periodic box."""
    for n in range(pos.shape[0]):
        for i in range(3):
            if pos[n, i] > boxsize:
                pos[n, i] -= boxsize
            elif pos[n, i] < 0:
                pos[n, i] += boxsize

    return pos


@jit(nopython=True, fastmath=True, boundscheck=False)
def delta2ncells(field):
    """
    Calculate the number of cells in `field` that are non-zero.
    """
    tot = 0
    imax, jmax, kmax = field.shape
    for i in range(imax):
        for j in range(jmax):
            for k in range(kmax):
                if field[i, j, k] > 0:
                    tot += 1
    return tot


def cartesian_to_radec(X):
    """
    Calculate the radial distance, RA [0, 360) deg and dec [-90, 90] deg.
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]

    dist = numpy.linalg.norm(X, axis=1)
    dec = numpy.arcsin(z / dist)
    ra = numpy.arctan2(y, x)
    ra[ra < 0] += 2 * numpy.pi

    ra *= 180 / numpy.pi
    dec *= 180 / numpy.pi

    return numpy.vstack([dist, ra, dec]).T


def radec_to_cartesian(X):
    """
    Calculate Cartesian coordinates from radial distance, RA [0, 360) deg  and
    dec [-90, 90] deg.
    """
    dist, ra, dec = X[:, 0], X[:, 1], X[:, 2]

    ra *= numpy.pi / 180
    dec *= numpy.pi / 180
    cdec = numpy.cos(dec)

    return numpy.vstack([
        dist * cdec * numpy.cos(ra),
        dist * cdec * numpy.sin(ra),
        dist * numpy.sin(dec)
        ]).T


def cosine_similarity(x, y):
    r"""
    Calculate the cosine similarity between two Cartesian vectors. Defined
    as :math:`\Sum_{i} x_i y_{i} / (|x| * |y|)`.

    Parameters
    ----------
    x : 1-dimensional array
        The first vector.
    y : 1- or 2-dimensional array
        The second vector. Can be 2-dimensional of shape `(n_samples, 3)`,
        in which case the calculation is broadcasted.

    Returns
    -------
    out : float or 1-dimensional array
    """
    if x.ndim != 1:
        raise ValueError("`x` must be a 1-dimensional array.")

    if y.ndim == 1:
        y = y.reshape(1, -1)

    out = numpy.sum(x * y, axis=1)
    out /= numpy.linalg.norm(x) * numpy.linalg.norm(y, axis=1)

    return out[0] if out.size == 1 else out


def real2redshift(pos, vel, observer_location, observer_velocity, box,
                  periodic_wrap=True, make_copy=True):
    r"""
    Convert real-space position to redshift space position.

    Parameters
    ----------
    pos : 2-dimensional array `(nsamples, 3)`
        Real-space Cartesian components in `Mpc / h`.
    vel : 2-dimensional array `(nsamples, 3)`
        Cartesian velocity in `km / s`.
    observer_location: 1-dimensional array `(3,)`
        Observer location in `Mpc / h`.
    observer_velocity: 1-dimensional array `(3,)`
        Observer velocity in `km / s`.
    box : py:class:`csiborg.read.CSiBORGBox`
        Box units.
    periodic_wrap : bool, optional
        Whether to wrap around the box, particles may be outside the default
        bounds once RSD is applied.
    make_copy : bool, optional
        Whether to make a copy of `pos` before modifying it.

    Returns
    -------
    pos : 2-dimensional array `(nsamples, 3)`
        Redshift-space Cartesian position in `Mpc / h`.
    """
    if make_copy:
        pos = numpy.copy(pos)
        vel = numpy.copy(vel)

    H0_inv = 1. / 100

    # Place the observer at the origin
    pos -= observer_location
    vel -= observer_velocity

    vr_dot = numpy.einsum('ij,ij->i', pos, vel)
    norm2 = numpy.einsum('ij,ij->i', pos, pos)

    pos *= (1 + H0_inv * vr_dot / norm2).reshape(-1, 1)

    # Place the observer back
    pos += observer_location
    if not make_copy:
        vel += observer_velocity

    if periodic_wrap:
        boxsize = box.box2mpc(1.)
        pos = periodic_wrap_grid(pos, boxsize)

    return pos


###############################################################################
#                           Statistics                                        #
###############################################################################


@jit(nopython=True, fastmath=True, boundscheck=False)
def number_counts(x, bin_edges):
    """
    Calculate counts of samples in bins.
    """
    out = numpy.full(bin_edges.size - 1, numpy.nan, dtype=numpy.float32)
    for i in range(bin_edges.size - 1):
        out[i] = numpy.sum((x >= bin_edges[i]) & (x < bin_edges[i + 1]))
    return out


def binned_statistic(x, y, left_edges, bin_width, statistic):
    """
    Calculate a binned statistic.
    """
    out = numpy.full(left_edges.size, numpy.nan, dtype=x.dtype)

    for i in range(left_edges.size):
        mask = (x >= left_edges[i]) & (x < left_edges[i] + bin_width)

        if numpy.any(mask):
            out[i] = statistic(y[mask])
    return out


def fprint(msg, verbose=True):
    """Print and flush a message with a timestamp."""
    if verbose:
        print(f"{datetime.now()}:   {msg}", flush=True)
