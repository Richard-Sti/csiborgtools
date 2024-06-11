# Copyright (C) 2023 Richard Stiskalek
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
Code to calculate the enclosed mass, momentum or bulk flow from various
radial-velocity based estimators (that may be flawed, be careful).
"""
import numpy as np
from numba import jit
from tqdm import tqdm


###############################################################################
#                 Enclosed mass at each distance from particles               #
###############################################################################


@jit(nopython=True, boundscheck=False)
def _enclosed_mass(rdist, mass, rmax, start_index):
    enclosed_mass = 0.

    for i in range(start_index, len(rdist)):
        if rdist[i] <= rmax:
            enclosed_mass += mass[i]
        else:
            break

    return enclosed_mass, i


def particles_enclosed_mass(rdist, mass, distances):
    """
    Calculate the enclosed mass at each distance from a set of particles. Note
    that the particles must be sorted by distance from the center of the box.

    Parameters
    ----------
    rdist : 1-dimensional array
        Sorted distance of particles from the center of the box.
    mass : 1-dimensional array
        Sorted mass of particles.
    distances : 1-dimensional array
        Distances at which to calculate the enclosed mass.

    Returns
    -------
    enclosed_mass : 1-dimensional array
        Enclosed mass at each distance.
    """
    enclosed_mass = np.full_like(distances, 0.)
    start_index = 0
    for i, dist in enumerate(distances):
        if i > 0:
            enclosed_mass[i] += enclosed_mass[i - 1]

        m, start_index = _enclosed_mass(rdist, mass, dist, start_index)
        enclosed_mass[i] += m

    return enclosed_mass


###############################################################################
#                     Enclosed mass from a density field                      #
###############################################################################


@jit(nopython=True)
def _cell_rdist(i, j, k, Ncells, boxsize):
    """Radial distance of the center of a cell from the center of the box."""
    xi = boxsize / Ncells * (i + 0.5) - boxsize / 2
    yi = boxsize / Ncells * (j + 0.5) - boxsize / 2
    zi = boxsize / Ncells * (k + 0.5) - boxsize / 2

    return (xi**2 + yi**2 + zi**2)**0.5


@jit(nopython=True, boundscheck=False)
def _field_enclosed_mass(field, rmax, boxsize):
    Ncells = field.shape[0]
    cell_volume = (1000 * boxsize / Ncells)**3

    mass = 0.
    volume = 0.
    for i in range(Ncells):
        for j in range(Ncells):
            for k in range(Ncells):
                if _cell_rdist(i, j, k, Ncells, boxsize) < rmax:
                    mass += field[i, j, k]
                    volume += 1.

    return mass * cell_volume, volume * cell_volume


def field_enclosed_mass(field, distances, boxsize):
    """
    Calculate the approximate enclosed mass within a given radius from a
    density field, counts the mass in cells and volume of cells whose
    centers are within the radius.

    Parameters
    ----------
    field : 3-dimensional array
        Density field in units of `h^2 Msun / kpc^3`.
    rmax : 1-dimensional array
        Radii to calculate the enclosed mass at in `Mpc / h`.
    boxsize : float
        Box size in `Mpc / h`.

    Returns
    -------
    enclosed_mass : 1-dimensional array
        Enclosed mass at each distance.
    enclosed_volume : 1-dimensional array
        Enclosed grid-like volume at each distance.
    """
    enclosed_mass = np.zeros_like(distances)
    enclosed_volume = np.zeros_like(distances)

    for i, dist in enumerate(tqdm(distances)):
        enclosed_mass[i], enclosed_volume[i] = _field_enclosed_mass(
            field, dist, boxsize)

    return enclosed_mass, enclosed_volume


###############################################################################
#              Enclosed momentum at each distance from particles              #
###############################################################################


@jit(nopython=True, boundscheck=False)
def _enclosed_momentum(rdist, mass, vel, rmax, start_index):
    bulk_momentum = np.zeros(3, dtype=rdist.dtype)

    for i in range(start_index, len(rdist)):
        if rdist[i] <= rmax:
            bulk_momentum += mass[i] * vel[i]
        else:
            break

    return bulk_momentum, i


def particles_enclosed_momentum(rdist, mass, vel, distances):
    """
    Calculate the enclosed momentum at each distance. Note that the particles
    must be sorted by distance from the center of the box.

    Parameters
    ----------
    rdist : 1-dimensional array
        Sorted distance of particles from the center of the box.
    mass : 1-dimensional array
        Sorted mass of particles.
    vel : 2-dimensional array
        Sorted velocity of particles.
    distances : 1-dimensional array
        Distances at which to calculate the enclosed momentum.

    Returns
    -------
    bulk_momentum : 2-dimensional array
        Enclosed momentum at each distance.
    """
    bulk_momentum = np.zeros((len(distances), 3))
    start_index = 0
    for i, dist in enumerate(distances):
        if i > 0:
            bulk_momentum[i] += bulk_momentum[i - 1]

        v, start_index = _enclosed_momentum(rdist, mass, vel, dist,
                                            start_index)
        bulk_momentum[i] += v

    return bulk_momentum


###############################################################################
#                         Bulk flow estimators                                #
###############################################################################


def bulkflow_peery2018(rdist, mass, pos, vel, distances, weights,
                       verbose=True):
    """
    Calculate the bulk flow from a set of particles using the estimator from
    Peery+2018. Supports either `1/r^2` or `constant` weights. Particles
    are assumed to be sorted by distance from the center of the box.

    Parameters
    ----------
    rdist : 1-dimensional array
        Sorted distance of particles from the center of the box.
    mass : 1-dimensional array
        Sorted mass of particles.
    pos : 2-dimensional array
        Sorted position of particles.
    vel : 2-dimensional array
        Sorted velocity of particles.
    distances : 1-dimensional array
        Distances at which to calculate the bulk flow.
    weights : str
        Weights to use in the estimator, either `1/r^2` or `constant`.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    bulk_flow : 2-dimensional array
    """
    # Select only the particles within the maximum distance to speed up the
    # calculation.
    if verbose:
        print("Selecting particles within the maximum distance...")
    kmax = np.searchsorted(rdist, np.max(distances))
    rdist = rdist[:kmax]
    mass = mass[:kmax]
    pos = pos[:kmax]
    vel = vel[:kmax]

    if verbose:
        print("Computing the cumulative quantities...")
    if weights == "1/r^2":
        cumulative_x = np.cumsum(mass[:, np.newaxis] * np.sum(vel * pos, axis=1)[:, np.newaxis] * pos / rdist[:, np.newaxis]**4, axis=0)  # noqa
        norm = lambda R: R**2  # noqa
    elif weights == "constant":
        cumulative_x = np.cumsum(mass[:, np.newaxis] * np.sum(vel * pos, axis=1)[:, np.newaxis] * pos / rdist[:, np.newaxis]**2, axis=0)  # noqa
        norm = lambda R: 3.  # noqa
    else:
        raise ValueError("Invalid weights.")
    cumulative_x /= np.cumsum(mass)[:, np.newaxis]

    B = np.zeros((len(distances), 3))
    for i in range(3):
        for j, R in enumerate(distances):
            k = np.searchsorted(rdist, R)
            B[j, i] = norm(R) * cumulative_x[k - 1, i]

    return B
