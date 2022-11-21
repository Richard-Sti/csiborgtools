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
from ..read import CombinedHaloCatalogue


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


class RealisationsMatcher:
    """
    A tool to match halos between IC realisations. Looks for halos 3D space
    neighbours in all remaining IC realisations that are within some mass
    range of it.


    Parameters
    ----------
    cats : :py:class`csiborgtools.read.CombinedHaloCatalogue`
        Combined halo catalogue to search.
    """
    _cats = None

    def __init__(self, cats):
        self.cats = cats

    @property
    def cats(self):
        """
        Combined catalogues.

        Returns
        -------
        cats : :py:class`csiborgtools.read.CombinedHaloCatalogue`
            Combined halo catalogue.
        """
        return self._cats

    @cats.setter
    def cats(self, cats):
        """
        Sets `cats`, ensures that this is a `CombinedHaloCatalogue` object.
        """
        if not isinstance(cats, CombinedHaloCatalogue):
            raise TypeError("`cats` must be of type `CombinedHaloCatalogue`.")
        self._cats = cats

    def cross_knn_position_single(self, n_sim, nmult=5, dlogmass=2):
        r"""
        Find all neighbours within :math:`n_{\rm mult} R_{200c}` of halos in
        the `nsim`th simulation. Also enforces that the neighbours'
        :math:`\log M_{200c}` be within `dlogmass` dex.

        Parameters
        ----------
        n_sim : int
            Index of an IC realisation in `self.cats` whose halos' neighbours
            in the remaining simulations to search for.
        nmult : float or int, optional
            Multiple of :math:`R_{200c}` within which to return neighbours. By
            default 5.
        dlogmass : float, optional
            Tolerance on mass logarithmic mass difference. By default 2 dex.

        Returns
        -------
        matches : composite array
            Array, indices are `(n_sims - 1, 2, n_halos, n_matches)`. The
            2nd axis is `index` of the neighbouring halo in its catalogue and
            `dist`, which is the 3D distance to the halo whose neighbours are
            searched.
        """
        N = self.cats.N  # Number of catalogues
        # R200c, M200c and positions of halos in `n_sim` IC realisation
        r200 = self.cats[n_sim]["r200"]
        logm200 = numpy.log10(self.cats[n_sim]["m200"])
        pos = self.cats[n_sim].positions

        matches = [None] * (N - 1)
        # Search for neighbours in the other simulations
        for count, i in enumerate([i for i in range(N) if i != n_sim]):
            dist, indxs = self.cats[i].radius_neigbours(pos, r200 * nmult)
            # Get rid of neighbors whose mass is too off
            for j, indx in enumerate(indxs):
                match_logm200 = numpy.log10(self.cats[i][indx]["m200"])
                mask = numpy.abs(match_logm200 - logm200[j]) < dlogmass
                dist[j] = dist[j][mask]
                indxs[j] = indx[mask]
            # Append as a composite array
            matches[count] = numpy.asarray([indxs, dist], dtype=object)

        return numpy.asarray(matches, dtype=object)

    def cross_knn_position_all(self, nmult=5, dlogmass=2):
        r"""
        Find all neighbours within :math:`n_{\rm mult} R_{200c}` of halos in
        all simulations listed in `self.cats`. Also enforces that the
        neighbours' :math:`\log M_{200c}` be within `dlogmass` dex.

        Parameters
        ----------
        nmult : float or int, optional
            Multiple of :math:`R_{200c}` within which to return neighbours. By
            default 5.
        dlogmass : float, optional
            Tolerance on mass logarithmic mass difference. By default 2 dex.

        Returns
        -------
        matches : list of composite arrays
            List whose length is `n_sims`. For description of its elements see
            `self.cross_knn_position_single(...)`.
        """
        N = self.cats.N  # Number of catalogues
        matches = [None] * N
        # Loop over each catalogue
        for i in range(N):
            matches[i] = self.cross_knn_position_single(i, nmult, dlogmass)
        return matches
