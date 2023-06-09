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
Tools for finding the nearest neighbours of reference simulation haloes from
cross simulations.
"""
import numpy


def find_neighbour(nsim0, cats):
    """
    Find the nearest neighbour of halos in `cat0` in `catx`.

    Parameters
    ----------
    nsim0 : int
        Index of the reference simulation.
    cats : dict
        Dictionary of halo catalogues. Keys must be the simulation indices.

    Returns
    -------
    dists : 2-dimensional array of shape `(nhalos, len(cats) - 1)`
        Distances to the nearest neighbour.
    cross_hindxs : 2-dimensional array of shape `(nhalos, len(cats) - 1)`
        Halo indices of the nearest neighbour.
    """
    cat0 = cats[nsim0]
    X = cat0.position(in_initial=False)
    shape = (X.shape[0], len(cats) - 1)
    dists = numpy.full(shape, numpy.nan, dtype=numpy.float16)
    cross_hindxs = numpy.full(shape, numpy.nan, dtype=numpy.int32)

    i = 0
    for nsimx, catx in cats.items():
        if nsimx == nsim0:
            continue
        dist, ind = catx.nearest_neighbours(X, radius=1, in_initial=False,
                                            knearest=True)
        dists[:, i] = dist.reshape(-1,)
        cross_hindxs[:, i] = catx["index"][ind.reshape(-1,)]
        i += 1

    return dists, cross_hindxs
