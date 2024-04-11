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

import numpy as np
from sklearn.neighbors import NearestNeighbors
from ..utils import cosine_similarity
from tqdm import trange
from scipy.stats import binned_statistic_2d


def pick_random_observer(rmax, boxsize, gen=None):
    """
    Pick a random observer within the box, such that the observer is at least
    `rmax` away from the box edges.

    Parameters
    ----------
    rmax : float
        The observer radius.
    boxsize : float
        The box size.
    gen : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    1-dimensional array
    """
    gen = np.random.default_rng() if gen is None else gen
    if boxsize - 2 * rmax < 0:
        raise ValueError("The box is too small for this observer radius.")

    return gen.uniform(rmax, boxsize - rmax, size=3)


def pick_random_los_cartesian(gen=None):
    """
    Pick a random line of sight in Cartesian coordinates, i.e. draw a random
    point on the unit sphere and convert it to Cartesian coordinates.

    Parameters
    ----------
    gen : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    1-dimensional array of shape (3,)
    """
    gen = np.random.default_rng() if gen is None else gen
    ra, dec = gen.uniform(0, 2 * np.pi), np.arccos(gen.uniform(-1, 1))

    cdec = np.cos(dec)
    return np.array([cdec * np.cos(ra),
                     cdec * np.sin(ra),
                     np.sin(dec)])


def count_angdist_per_dist_log_mass(cartesian_pos, log_mass, boxsize,
                                    rdist_bin_edges, log_mass_bin_edges,
                                    nrepeat=1000):
    """
    Count the minimum angular distance between a random observer and a random
    line of sight as a function of distance and mass. The distance and mass
    dependence is evaluated in bins and this function returns the minimum
    angular distance in each bin for each repeat.

    Parameters
    ----------
    cartesian_pos : 2-dimensional array of shape `(n, 3)`
        The Cartesian positions of the objects.
    log_mass : 1-dimensional array of shape `(n,)`
        The log mass of the objects.
    boxsize : float
        The box size.
    rdist_bin_edges : 1-dimensional array
        The radial distance bin edges.
    log_mass_bin_edges : 1-dimensional array
        The log mass bin edges.
    nrepeat : int, optional
        The number of repeats, recommended to be a very high number.

    Returns
    -------
    3-dimensional array of shape `(len(rdist_bin_edges) - 1,
        len(log_mass_bin_edges) - 1, nrepeat)`
    """
    # Calculate the kNN for the halo positions
    knn = NearestNeighbors()
    knn.fit(cartesian_pos)

    gen = np.random.default_rng()
    rmax = rdist_bin_edges[-1]
    min_angdist = np.zeros((len(rdist_bin_edges)-1, len(log_mass_bin_edges)-1,
                            nrepeat))

    for i in trange(nrepeat, desc="Sampling"):
        # Sample a random observer and line of sight
        obs = pick_random_observer(rmax, boxsize, gen).reshape(1, -1)
        los = pick_random_los_cartesian(gen)

        # Select all haloes within the observer radius
        rdist, ks = knn.radius_neighbors(obs, rmax, return_distance=True)
        rdist, ks = rdist[0], ks[0]

        # Calculate the haloes' angular distance from the line of sight
        angdist = np.arccos(cosine_similarity(los, cartesian_pos[ks] - obs))

        # Within each distance x mass bin, find the minimum angular distance
        min_angdist[..., i] = binned_statistic_2d(
            rdist, log_mass[ks], angdist, statistic="min",
            bins=[rdist_bin_edges, log_mass_bin_edges], )[0]

    return min_angdist
