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
from tqdm import (tqdm, trange)
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
    # NOTE add later
#    dtype : dtype, optional
#        Output precision. By default `numpy.float32`.
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

    def search_sim_indices(self, n_sim):
        """
        Return indices of all other IC realisations but of `n_sim`.

        Parameters
        ----------
        n_sim : int
            IC realisation index of `self.cats` to be skipped.

        Returns
        -------
        indxs : list of int
            The remaining IC indices.
        """
        return [i for i in range(self.cats.N) if i != n_sim]

    def _check_masskind(self, mass_kind):
        """Check that `mass_kind` is a valid key."""
        if mass_kind not in self.cats[0].keys:
            raise ValueError("Invalid mass kind `{}`.".format(mass_kind))

    @staticmethod
    def _cat2clump_mapping(cat_indxs, clump_indxs):
        """
        Create a mapping from a catalogue array index to a clump array index.

        Parameters
        ----------
        cat_indxs : 1-dimensional array
            Clump indices in the catalogue array.
        clump_indxs : 1-dimensional array
            Clump indices in the clump array.

        Returns
        -------
        mapping : 1-dimensional array
            Mapping. The array indices match catalogue array and values are
            array positions in the clump array.
        """
        mapping = numpy.full(cat_indxs.size, numpy.nan, dtype=int)
        __, ind1, ind2 = numpy.intersect1d(clump_indxs, cat_indxs,
                                           return_indices=True)
        mapping[ind2] = ind1
        return mapping

    def cross_knn_position_single(self, n_sim, nmult=5, dlogmass=None,
                                  mass_kind="totpartmass", init_dist=False,
                                  overlap=False, verbose=True):
        r"""
        Find all neighbours within :math:`n_{\rm mult} R_{200c}` of halos in
        the `nsim`th simulation. Also enforces that the neighbours'
        :math:`\log M / M_\odot` be within `dlogmass` dex.

        Parameters
        ----------
        n_sim : int
            Index of an IC realisation in `self.cats` whose halos' neighbours
            in the remaining simulations to search for.
        nmult : float or int, optional
            Multiple of :math:`R_{200c}` within which to return neighbours. By
            default 5.
        dlogmass : float, optional
            Tolerance on mass logarithmic mass difference. By default `None`.
        mass_kind : str, optional
            The mass kind whose similarity is to be checked. Must be a valid
            catalogue key. By default `totpartmass`, i.e. the total particle
            mass associated with a halo.
        init_dist : bool, optional
            Whether to calculate separation of the initial CMs. By default
            `False`.
        overlap : bool, optional
            Whether to calculate overlap between clumps in the initial
            snapshot. By default `False`. Note that this operation is
            substantially slower.
        verbose : bool, optional
            Iterator verbosity flag. By default `True`.

        Returns
        -------
        matches : composite array
            Array, indices are `(n_sims - 1, 4, n_halos, n_matches)`. The
            2nd axis is `index` of the neighbouring halo in its catalogue,
            `dist` is the 3D distance to the halo whose neighbours are
            searched, `dist0` is the separation of the initial CMs and
            `overlap` is the overlap over the initial clumps, all respectively.
            The latter two are calculated only if `init_dist` or `overlap` is
            `True`.

        TODO:
        - [ ] Precalculate the mapping from halo index to clump array position
        """
        self._check_masskind(mass_kind)
        # Radius, mass and positions of halos in `n_sim` IC realisation
        logmass = numpy.log10(self.cats[n_sim][mass_kind])
        R = self.cats[n_sim]["r200"]
        pos = self.cats[n_sim].positions
        if init_dist:
            pos0 = self.cats[n_sim].positions0  # These are CM positions
        if overlap:
            if verbose:
                print("Loading initial clump particles for `n_sim = {}`."
                      .format(n_sim))
            # Grab a paths object. What it is set to is unimportant
            paths = self.cats[0].paths
            with open(paths.clump0_path(self.cats.n_sims[n_sim]), "rb") as f:
                clumps0 = numpy.load(f, allow_pickle=True)
            overlapper = ParticleOverlap()
            cat2clumps0 = self._cat2clump_mapping(self.cats[n_sim]["index"],
                                                  clumps0["ID"])

        matches = [None] * (self.cats.N - 1)
        # Verbose iterator
        if verbose:
            iters = enumerate(tqdm(self.search_sim_indices(n_sim)))
        else:
            iters = enumerate(self.search_sim_indices(n_sim))
        iters = enumerate(self.search_sim_indices(n_sim))
        # Search for neighbours in the other simulations
        for count, i in iters:
            dist, indxs = self.cats[i].radius_neigbours(pos, R * nmult)
            # Get rid of neighbors whose mass is too off
            if dlogmass is not None:
                for j, indx in enumerate(indxs):
                    match_logmass = numpy.log10(self.cats[i][mass_kind][indx])
                    mask = numpy.abs(match_logmass - logmass[j]) < dlogmass
                    dist[j] = dist[j][mask]
                    indxs[j] = indx[mask]
            # Find distance to the between the initial CM
            dist0 = [numpy.asanyarray([], dtype=numpy.float64)] * dist.size
            if init_dist:
                with_neigbours = numpy.where([ii.size > 0 for ii in indxs])[0]
                # Fill the pre-allocated array on positions with neighbours
                for k in with_neigbours:
                    dist0[k] = numpy.linalg.norm(
                        pos0[k] - self.cats[i].positions0[indxs[k]], axis=1)

            # Calculate the initial snapshot overlap
            cross = [numpy.asanyarray([], dtype=numpy.float64)] * dist.size
            if overlap:
                if verbose:
                    print("Loading initial clump particles for `n_sim = {}` "
                          "to compare against `n_sim = {}`.".format(i, n_sim))
                with open(paths.clump0_path(self.cats.n_sims[i]), 'rb') as f:
                    clumpsx = numpy.load(f, allow_pickle=True)
                cat2clumpsx = self._cat2clump_mapping(self.cats[i]["index"],
                                                      clumpsx["ID"])

                # Loop only over halos that have neighbours
                with_neigbours = numpy.where([ii.size > 0 for ii in indxs])[0]
                for k in tqdm(with_neigbours) if verbose else with_neigbours:
                    # Find which clump matches index of this halo from cat
                    match0 = cat2clumps0[k]

                    # Get the clump and pre-calculate its cell assignment
                    cl0 = clumps0["clump"][match0]
                    cl0_cells = overlapper.assign_to_cell(
                        *(cl0[p] for p in ('x', 'y', 'z')))
                    dint = numpy.full(indxs[k].size, numpy.nan, numpy.float64)

                    # Loop over the ones we cross-correlate with
                    for ii, ind in enumerate(indxs[k]):
                        # Again which cross clump to this index
                        matchx = cat2clumpsx[ind]
                        dint[ii] = overlapper.mass_overlap(
                            cl0, clumpsx["clump"][matchx], cl0_cells)

                    cross[k] = dint

            # Append as a composite array
            matches[count] = numpy.asarray(
                [indxs, dist, dist0, cross], dtype=object)

        return numpy.asarray(matches, dtype=object)

    def cross_knn_position_all(self, nmult=5, dlogmass=None,
                               mass_kind="totpartmass", init_dist=False,
                               overlap=False, verbose=True):
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
            Tolerance on mass logarithmic mass difference. By default `None`.
        mass_kind : str, optional
            The mass kind whose similarity is to be checked. Must be a valid
            catalogue key. By default `totpartmass`, i.e. the total particle
            mass associated with a halo.
        init_dist : bool, optional
            Whether to calculate separation of the initial CMs. By default
            `False`.
        overlap : bool, optional
            Whether to calculate overlap between clumps in the initial
            snapshot. By default `False`. Note that this operation is
            substantially slower.
        verbose : bool, optional
            Iterator verbosity flag. By default `True`.

        Returns
        -------
        matches : list of composite arrays
            List whose length is `n_sims`. For description of its elements see
            `self.cross_knn_position_single(...)`.
        """
        N = self.cats.N  # Number of catalogues
        matches = [None] * N
        # Loop over each catalogue
        for i in trange(N) if verbose else range(N):
            matches[i] = self.cross_knn_position_single(
                i, nmult, dlogmass, mass_kind=mass_kind, init_dist=init_dist,
                overlap=overlap, verbose=verbose)
        return matches


###############################################################################
#                           Matching statistics                               #
###############################################################################


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
        The cosine similarity. If y is 1-dimensinal returns only a float.
    """
    # Quick check of dimensions
    if x.ndim != 1:
        raise ValueError("`x` must be a 1-dimensional array.")
    y = y.reshape(-1, 3) if y.ndim == 1 else y

    out = numpy.sum(x * y, axis=1)
    out /= numpy.linalg.norm(x) * numpy.linalg.norm(y, axis=1)

    if out.size == 1:
        return out[0]
    return out


class ParticleOverlap:
    """
    TODO:
    - [ ] Class documentation
    """
    _bins = None

    def __init__(self, bins=None):
        if bins is None:
            dx = 1 / 2**11
            bins = numpy.arange(0, 1 + dx, dx)
        self.bins = bins

    @property
    def bins(self):
        """
        The grid spacing. Assumed to be equal for all three dimensions. Units
        ought to match the requested coordinates.

        Returns
        -------
        bins : 1-dimensional array
        """
        return self._bins

    @bins.setter
    def bins(self, bins):
        """Sets `bins`."""
        bins = numpy.asarray(bins) if isinstance(bins, list) else bins
        assert bins.ndim == 1, "`bins` must be a 1-dimensional array."
        self._bins = bins

    def assign_to_cell(self, x, y, z):
        """
        Assign particles specified by coordinates `x`, `y`, and `z` to grid
        cells.

        Parameters
        ----------
        x, y, z : 1-dimensional arrays
            Positions of particles in the box.

        Returns
        -------
        cells : 1-dimensional array
            Cell ID of each particle.
        """
        assert x.ndim == 1 and x.size == y.size == z.size
        xbin = numpy.digitize(x, self.bins)
        ybin = numpy.digitize(y, self.bins)
        zbin = numpy.digitize(z, self.bins)
        N = self.bins.size

        return xbin + ybin * N + zbin * N**2

    def mass_overlap(self, clump1, clump2, cells1=None):
        r"""
        Calculate the particle, mass-weighted overlap between two halos.
        Defined as

        ..math::
            (M_{u,1} + M_{u,2}) / (M_1 + M_2),

        where :math:`M_{u, 1}` is the mass of particles of the first halo in
        cells that are also present in the second halo and :math:`M_1` is the
        total particle mass of the first halo.

        Parameters
        ----------
        clump1, clump2 : structured arrays
            Structured arrays corresponding to the two clumps. Should contain
            keys `x`, `y`, `z` and `M`.
        cells1 : 1-dimensional array, optional
            Optionlaly precomputed cells of `clump1`. Be careful when using
            this to ensure it matches `clump1`.

        Returns
        -------
        overlap : float
        """
        # 1-dimensional cell ID of each particle in clump1 and clump2
        if cells1 is None:
            cells1 = self.assign_to_cell(*[clump1[p] for p in ('x', 'y', 'z')])
        cells2 = self.assign_to_cell(*[clump2[p] for p in ('x', 'y', 'z')])
        # Elementwise cells1 in cells2 and vice versa
        m1 = numpy.isin(cells1, cells2)
        m2 = numpy.isin(cells2, cells1)
        # Summed shared mass and the total
        interp = numpy.sum(clump1["M"][m1]) + numpy.sum(clump2["M"][m2])
        mtot = numpy.sum(clump1["M"]) + numpy.sum(clump2["M"])

        return interp / mtot
