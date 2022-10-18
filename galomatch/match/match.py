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

import numpy
from tqdm import tqdm
from astropy.coordinates import SkyCoord


def brute_spatial_separation(c1, c2, angular=False, N=None, verbose=False):
    """
    Calculate for each point in `c1` the `N` closest points in `c2`.

    Parameters
    ----------
    c1 : `astropy.coordinates.SkyCoord`
        Coordinates of the first set of points.
    c2 : `astropy.coordinates.SkyCoord`
        Coordinates of the second set of points.
    angular : bool, optional
        Whether to calculate angular separation or 3D separation. By default
        `False` and 3D separation is calculated.
    N : int, optional
        Number of closest points in `c2` to each object in `c1` to return.
    verbose : bool, optional
        Verbosity flag. By default `False`.

    Returns
    -------
    sep : 1-dimensional array
        Separation of each object in `c1` to `N` closest objects in `c2`. The
        array shape is `(c1.size, N)`. Separation is in units of `c1`.
    indxs : 1-dimensional array
        Indexes of the closest objects in `c2` for each object in `c1`. The
        array shape is `(c1.size, N)`.
    """
    if not (isinstance(c1, SkyCoord) and isinstance(c2, SkyCoord)):
        raise TypeError("`c1` & `c2` must be `astropy.coordinates.SkyCoord`.")
    N1 = c1.size
    N2 = c2.size if N is None else N

    # Pre-allocate arrays
    sep = numpy.full((N1, N2), numpy.nan)
    indxs = numpy.full((N1, N2), numpy.nan, dtype=int)
    iters = tqdm(range(N1)) if verbose else range(N1)
    for i in iters:
        if angular:
            dist = c1[i].separation(c2).value
        else:
            dist = c1[i].separation_3d(c2).value
        # Sort the distances
        sort = numpy.argsort(dist)[:N2]
        indxs[i, :] = sort
        sep[i, :] = dist[sort]

    return sep, indxs
