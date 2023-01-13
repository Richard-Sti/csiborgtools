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
from scipy.ndimage import gaussian_filter
from tqdm import (tqdm, trange)
from astropy.coordinates import SkyCoord
from numba import jit
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
                                  overlap=False, overlapper_kwargs={},
                                  verbose=True):
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
        overlapper_kwargs : dict, optional
            Keyword arguments passed to `ParticleOverlapper`.
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
            overlapper = ParticleOverlap(**overlapper_kwargs)
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
                    dint = numpy.full(indxs[k].size, numpy.nan, numpy.float64)

                    # Loop over the ones we cross-correlate with
                    for ii, ind in enumerate(indxs[k]):
                        # Again which cross clump to this index
                        matchx = cat2clumpsx[ind]
                        dint[ii] = overlapper(cl0, clumpsx["clump"][matchx])

                    cross[k] = dint

            # Append as a composite array
            matches[count] = numpy.asarray(
                [indxs, dist, dist0, cross], dtype=object)

        return numpy.asarray(matches, dtype=object)

    def cross_knn_position_all(self, nmult=5, dlogmass=None,
                               mass_kind="totpartmass", init_dist=False,
                               overlap=False, overlapper_kwargs={},
                               verbose=True):
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
        overlapper_kwargs : dict, optional
            Keyword arguments passed to `ParticleOverlapper`.
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
                overlap=overlap, overlapper_kwargs=overlapper_kwargs,
                verbose=verbose)
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
    r"""
    Class to calculate overlap between two halos from different simulations.
    The density field calculation is based on the nearest grid position
    particle assignment scheme, with optional additional Gaussian smoothing.

    Parameters
    ----------
    inv_clength : float, optional
        Inverse cell length in box units. By default :math:`2^11`, which
        matches the initial RAMSES grid resolution.
    smooth_scale : float or integer, optional
        Optional Gaussian smoothing scale to by applied to the fields. By
        default no smoothing is applied. Otherwise the scale is to be
        specified in the number of cells (i.e. in units of `self.cellsize`).
    """
    _inv_clength = None
    _smooth_scale = None
    _clength = None
    _ncells = None

    def __init__(self, inv_clength=2**11, smooth_scale=None):
        self.inv_clength = inv_clength
        self.smooth_scale = smooth_scale

        self._clength = 1 / self.inv_clength
        self._ncells = int(inv_clength)

    @property
    def inv_clength(self):
        """
        Inverse cell length.

        Returns
        -------
        inv_clength : float
        """
        return self._inv_clength

    @inv_clength.setter
    def inv_clength(self, inv_clength):
        """Sets `inv_clength`."""
        assert inv_clength > 0, "`inv_clength` must be positive."
        assert isinstance(inv_clength, int), "`inv_clength` must be integer."
        self._inv_clength = int(inv_clength)

    @property
    def smooth_scale(self):
        """
        The smoothing scale in units of `self.cellsize`. If not set `None`.

        Returns
        -------
        smooth_scale : int or float
        """
        return self._smooth_scale

    @smooth_scale.setter
    def smooth_scale(self, smooth_scale):
        """Sets `smooth_scale`."""
        if smooth_scale is not None:
            assert smooth_scale > 0
            self._smooth_scale = smooth_scale

    def pos2cell(self, pos):
        """
        Convert position to cell number.

        Parameters
        ----------
        pos : 1-dimensional array

        Returns
        -------
        cells : 1-dimensional array
        """
        return numpy.floor(pos * self.inv_clength).astype(int)

    def make_delta(self, clump, subbox=False):
        """
        Calculate a NGP density field of a halo on a cubic grid.

        Parameters
        ----------
        clump: structurered arrays
            Clump structured array, keys must include `x`, `y`, `z` and `M`.
        subbox : bool, optional
            Whether to calculate the density field on a grid strictly enclosing
            the clump.

        Returns
        -------
        delta : 3-dimensional array
        """
        coords = ('x', 'y', 'z')
        xcell, ycell, zcell = (self.pos2cell(clump[p]) for p in coords)
        if subbox:
            # Shift the box so that each non-zero grid cell is 0th
            xcell -= numpy.min(xcell)
            ycell -= numpy.min(ycell)
            zcell -= numpy.min(zcell)
            ncells = max(*(numpy.max(p) for p in (xcell, ycell, zcell))) + 1
        else:
            ncells = self.inv_clength

        # Preallocate and fill the array
        delta = numpy.zeros((ncells,) * 3, dtype=numpy.float32)
        fill_delta(delta, xcell, ycell, zcell, clump['M'])

        if self.smooth_scale is not None:
            gaussian_filter(delta, self.smooth_scale, output=delta)
        return delta

    def make_deltas(self, clump1, clump2):
        """
        Calculate a NGP density fields of two halos on a grid that encloses
        them both.

        Parameters
        ----------
        clump1, clump2 : structurered arrays
            Particle structured array of the two clumps. Keys must include `x`,
            `y`, `z` and `M`.

        Returns
        -------
        delta1, delta2 : 3-dimensional arrays
            Density arrays of `clump1` and `clump2`, respectively.
        """
        coords = ('x', 'y', 'z')
        xcell1, ycell1, zcell1 = (self.pos2cell(clump1[p]) for p in coords)
        xcell2, ycell2, zcell2 = (self.pos2cell(clump2[p]) for p in coords)

        # Minimum cell number of the two halos along each dimension
        xmin = min(numpy.min(xcell1), numpy.min(xcell2))
        ymin = min(numpy.min(ycell1), numpy.min(ycell2))
        zmin = min(numpy.min(zcell1), numpy.min(zcell2))
        # Maximum cell number of the two halos along each dimension
        xmax = max(numpy.max(xcell1), numpy.max(xcell2))
        ymax = max(numpy.max(ycell1), numpy.max(ycell2))
        zmax = max(numpy.max(zcell1), numpy.max(zcell2))

        # Number of cells is the maximum + 1
        ncells = max(xmax - xmin, ymax - ymin, zmax - zmin) + 1

        # Shift the box so that the first non-zero grid cell is 0th
        xcell1 -= xmin
        xcell2 -= xmin
        ycell1 -= ymin
        ycell2 -= ymin
        zcell1 -= zmin
        zcell2 -= zmin

        # Preallocate and fill the array
        delta1 = numpy.zeros((ncells,)*3, dtype=numpy.float32)
        fill_delta(delta1, xcell1, ycell1, zcell1, clump1['M'])
        delta2 = numpy.zeros((ncells,)*3, dtype=numpy.float32)
        fill_delta(delta2, xcell2, ycell2, zcell2, clump2['M'])

        if self.smooth_scale is not None:
            gaussian_filter(delta1, self.smooth_scale, output=delta1)
            gaussian_filter(delta2, self.smooth_scale, output=delta2)
        return delta1, delta2

    @staticmethod
    def overlap(delta1, delta2):
        r"""
        Overlap between two density grids.

        Parameters
        ----------
        delta1, delta2 : 3-dimensional arrays
            Density arrays.

        Returns
        -------
        overlap : float
        """
        mass1 = numpy.sum(delta1)
        mass2 = numpy.sum(delta2)
        # Cells where both fields are > 0
        mask = (delta1 > 0) & (delta2 > 0)
        # Note the factor of 0.5 to avoid double counting
        intersect = 0.5 * numpy.sum(delta1[mask] + delta2[mask])
        return intersect / (mass1 + mass2 - intersect)

    def __call__(self, clump1, clump2):
        """
        Calculate overlap between `clump1` and `clump2`. See
        `self.overlap(...)` and `self.make_deltas(...)` for further
        information.

        Parameters
        ----------
        clump1, clump2 : structurered arrays
            Structured arrays containing the particles of a given clump. Keys
            must include `x`, `y`, `z` and `M`.

        Returns
        -------
        overlap : float
        """
        delta1, delta2 = self.make_deltas(clump1, clump2)
        return self.overlap(delta1, delta2)


@jit(nopython=True)
def fill_delta(delta, xcell, ycell, zcell, weights):
    """
    Fill array delta at the specified indices with their weights.

    Parameters
    ----------
    delta : 3-dimensional array
        Grid to be filled with weights.
    xcell, ycell, zcell : 1-dimensional arrays
        Indices where to assign `weights`.
    weights : 1-dimensional arrays
        Particle mass.

    Returns
    -------
    None
    """
    for i in range(xcell.size):
        delta[xcell[i], ycell[i], zcell[i]] += weights[i]
