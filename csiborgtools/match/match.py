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
from ..read import (CombinedHaloCatalogue, concatenate_clumps)


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
                                  mass_kind="totpartmass", overlap=False,
                                  overlapper_kwargs={}, select_initial=True,
                                  verbose=True):
        r"""
        Find all neighbours within a multiple of either :math:`R_{\rm init}`
        (distance at :math:`z = 70`) or :math:`R_{200c}` (distance at
        :math:`z = 0`) of halos in the `nsim`th simulation. Enforces that the
        neighbours' are similar in mass up to `dlogmass` dex.

        Parameters
        ----------
        n_sim : int
            Index of an IC realisation in `self.cats` whose halos' neighbours
            in the remaining simulations to search for.
        nmult : float or int, optional
            Multiple of :math:`R_{\rm init}` within which to return neighbours.
            By default 5.
        dlogmass : float, optional
            Tolerance on mass logarithmic mass difference. By default `None`.
        mass_kind : str, optional
            The mass kind whose similarity is to be checked. Must be a valid
            catalogue key. By default `totpartmass`, i.e. the total particle
            mass associated with a halo.
        overlap : bool, optional
            Whether to calculate overlap between clumps in the initial
            snapshot. By default `False`. This operation is slow.
        overlapper_kwargs : dict, optional
            Keyword arguments passed to `ParticleOverlapper`.
        select_initial : bool, optional
            Whether to select nearest neighbour at the initial or final
            snapshot. By default `True`, i.e. at the initial snapshot.
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
        # Halo properties of this simulation
        logmass = numpy.log10(self.cats[n_sim][mass_kind])
        pos = self.cats[n_sim].positions        # Grav potential minimum
        pos0 = self.cats[n_sim].positions0      # CM positions
        if select_initial:
            R = self.cats[n_sim].init_radius    # Approximate initial size
        else:
            R = self.cats[n_sim]["r200"]        # R200c at z = 0
        R *= nmult                              # Search radius

        if overlap:
            if verbose:
                print("Loading initial clump particles for `n_sim = {}`."
                      .format(n_sim), flush=True)
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
        # Search for neighbours in the other simulations at z = 70
        for count, i in iters:
            if select_initial:
                dist0, indxs = self.cats[i].radius_initial_neigbours(pos0, R)
            else:
                # Will switch dist0 <-> dist at the end
                dist0, indxs = self.cats[i].radius_neigbours(pos, R)

            # Get rid of neighbors whose mass is too off
            if dlogmass is not None:
                for j, indx in enumerate(indxs):
                    match_logmass = numpy.log10(self.cats[i][mass_kind][indx])
                    mask = numpy.abs(match_logmass - logmass[j]) < dlogmass
                    dist0[j] = dist0[j][mask]
                    indxs[j] = indx[mask]

            # Find the distance at z = 0 (or z = 70 dep. on `search_initial``)
            dist = [numpy.asanyarray([], dtype=numpy.float64)] * dist0.size
            with_neigbours = numpy.where([ii.size > 0 for ii in indxs])[0]
            # Fill the pre-allocated array on positions with neighbours
            for k in with_neigbours:
                if select_initial:
                    dist[k] = numpy.linalg.norm(
                        pos[k] - self.cats[i].positions[indxs[k]], axis=1)
                else:
                    dist[k] = numpy.linalg.norm(
                        pos0[k] - self.cats[i].positions0[indxs[k]], axis=1)

            # Calculate the initial snapshot overlap
            cross = [numpy.asanyarray([], dtype=numpy.float64)] * dist0.size
            if overlap:
                if verbose:
                    print("Loading initial clump particles for `n_sim = {}` "
                          "to compare against `n_sim = {}`.".format(i, n_sim),
                          flush=True)
                with open(paths.clump0_path(self.cats.n_sims[i]), 'rb') as f:
                    clumpsx = numpy.load(f, allow_pickle=True)

                # Calculate the particle field
                if verbose:
                    print("Loading and smoothing the crossed total field.",
                          flush=True)
                particles = concatenate_clumps(clumpsx)
                delta = overlapper.make_delta(particles, to_smooth=False)
                delta = overlapper.smooth_highres(delta)

                cat2clumpsx = self._cat2clump_mapping(self.cats[i]["index"],
                                                      clumpsx["ID"])
                # Loop only over halos that have neighbours
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
                        dint[ii] = overlapper(cl0, clumpsx["clump"][matchx],
                                              delta)

                    cross[k] = dint

            # Append as a composite array. Flip dist order if not select_init
            if select_initial:
                matches[count] = numpy.asarray(
                    [indxs, dist, dist0, cross], dtype=object)
            else:
                matches[count] = numpy.asarray(
                    [indxs, dist0, dist, cross], dtype=object)

        return numpy.asarray(matches, dtype=object)

    def cross_knn_position_all(self, nmult=5, dlogmass=None,
                               mass_kind="totpartmass", init_dist=False,
                               overlap=False, overlapper_kwargs={},
                               select_initial=True, verbose=True):
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
        select_initial : bool, optional
            Whether to select nearest neighbour at the initial or final
            snapshot. By default `True`, i.e. at the initial snapshot.
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
                select_initial=select_initial, verbose=verbose)
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
    nshift : int, optional
        Number of cells by which to shift the subbox from the outside-most
        cell containing a particle. By default 5.
    smooth_scale : float or integer, optional
        Optional Gaussian smoothing scale to by applied to the fields. By
        default no smoothing is applied. Otherwise the scale is to be
        specified in the number of cells (i.e. in units of `self.cellsize`).
    """
    _inv_clength = None
    _smooth_scale = None
    _clength = None
    _nshift = None

    def __init__(self, inv_clength=2**11, smooth_scale=None, nshift=5):
        self.inv_clength = inv_clength
        self.smooth_scale = smooth_scale
        self.nshift = nshift

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
        # Also set the inverse and number of cells
        self._clength = 1 / self.inv_clength

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
        if smooth_scale is None:
            self._smooth_scale = None
        else:
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

    def smooth_highres(self, delta):
        """
        Smooth the central region of a full box density field. Note that if
        `self.smooth_scale` is `None` then quietly exits the function.

        Parameters
        ----------
        delta : 3-dimensional array

        Returns
        -------
        smooth_delta : 3-dimensional arrray
        """
        if self.smooth_scale is None:
            return delta
        msg = "Shape of `delta` must match the entire box."
        assert delta.shape == (self._inv_clength,)*3, msg

        # Subselect only the high-resolution region
        start = self._inv_clength // 4
        end = start * 3
        highres = delta[start:end, start:end, start:end]
        # Smoothen it
        gaussian_filter(highres, self.smooth_scale, output=highres)
        # Put things back into the original array
        delta[start:end, start:end, start:end] = highres
        return delta

    def make_delta(self, clump, subbox=False, to_smooth=True):
        """
        Calculate a NGP density field of a halo on a cubic grid.

        Parameters
        ----------
        clump: structurered arrays
            Clump structured array, keys must include `x`, `y`, `z` and `M`.
        subbox : bool, optional
            Whether to calculate the density field on a grid strictly enclosing
            the clump.
        to_smooth : bool, optional
            Explicit control over whether to smooth. By default `True`.

        Returns
        -------
        delta : 3-dimensional array
        """
        coords = ('x', 'y', 'z')
        xcell, ycell, zcell = (self.pos2cell(clump[p]) for p in coords)
        if subbox:
            # Shift the box so that each non-zero grid cell is 0th
            xcell -= max(numpy.min(xcell) - self.nshift, 0)
            ycell -= max(numpy.min(ycell) - self.nshift, 0)
            zcell -= max(numpy.min(zcell) - self.nshift, 0)

            ncells = max(*(numpy.max(p) + self.nshift
                           for p in (xcell, ycell, zcell)))
            ncells += 1  # Bump up by one to get NUMBER of cells
            ncells = min(ncells, self.inv_clength)
        else:
            ncells = self.inv_clength

        # Preallocate and fill the array
        delta = numpy.zeros((ncells,) * 3, dtype=numpy.float32)
        fill_delta(delta, xcell, ycell, zcell, clump['M'])

        if to_smooth and self.smooth_scale is not None:
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
        cellmins : len-3 tuple
            Tuple of left-most cell ID in the full box.
        """
        coords = ('x', 'y', 'z')
        xcell1, ycell1, zcell1 = (self.pos2cell(clump1[p]) for p in coords)
        xcell2, ycell2, zcell2 = (self.pos2cell(clump2[p]) for p in coords)

#    if any(clumps[0][0].dtype[p].char in numpy.typecodes["AllInteger"]

        # Minimum cell number of the two halos along each dimension
        xmin = min(numpy.min(xcell1), numpy.min(xcell2)) - self.nshift
        ymin = min(numpy.min(ycell1), numpy.min(ycell2)) - self.nshift
        zmin = min(numpy.min(zcell1), numpy.min(zcell2)) - self.nshift
        xmin, ymin, zmin = max(xmin, 0), max(ymin, 0), max(zmin, 0)
        cellmins = (xmin, ymin, zmin)
        # Maximum cell number of the two halos along each dimension
        xmax = max(numpy.max(xcell1), numpy.max(xcell2))
        ymax = max(numpy.max(ycell1), numpy.max(ycell2))
        zmax = max(numpy.max(zcell1), numpy.max(zcell2))

        # Number of cells is the maximum + 1
        ncells = max(xmax - xmin, ymax - ymin, zmax - zmin) + self.nshift
        ncells += 1
        ncells = min(ncells, self.inv_clength)

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
        return delta1, delta2, cellmins

    @staticmethod
    def overlap(delta1, delta2, cellmins, delta2_full):
        r"""
        Overlap between two clumps whose density fields are evaluated on the
        same grid.

        Parameters
        ----------
        delta1, delta2 : 3-dimensional arrays
            Clumps density fields.
        cellmins : len-3 tuple
            Tuple of left-most cell ID in the full box.
        delta2_full : 3-dimensional array
            Density field of the whole box calculated with particles assigned
            to halos at zero redshift.

        Returns
        -------
        overlap : float
        """
        return _calculate_overlap(delta1, delta2, cellmins, delta2_full)

    def __call__(self, clump1, clump2, delta2_full):
        """
        Calculate overlap between `clump1` and `clump2`. See
        `self.overlap(...)` and `self.make_deltas(...)` for further
        information.

        Parameters
        ----------
        clump1, clump2 : structurered arrays
            Structured arrays containing the particles of a given clump. Keys
            must include `x`, `y`, `z` and `M`.
        cellmins : len-3 tuple
            Tuple of left-most cell ID in the full box.
        delta2_full : 3-dimensional array
            Density field of the whole box calculated with particles assigned
            to halos at zero redshift.

        Returns
        -------
        overlap : float
        """
        delta1, delta2, cellmins = self.make_deltas(clump1, clump2)
        return _calculate_overlap(delta1, delta2, cellmins, delta2_full)


@jit(nopython=True)
def fill_delta(delta, xcell, ycell, zcell, weights):
    """
    Fill array delta at the specified indices with their weights. This is a JIT
    implementation.

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


@jit(nopython=True)
def _calculate_overlap(delta1, delta2, cellmins, delta2_full):
    r"""
    Overlap between two clumps whose density fields are evaluated on the
    same grid. This is a JIT implementation, hence it is outside of the main
    class.

    Parameters
    ----------
    delta1, delta2 : 3-dimensional arrays
        Clumps density fields.
    cellmins : len-3 tuple
        Tuple of left-most cell ID in the full box.
    delta2_full : 3-dimensional array
        Density field of the whole box calculated with particles assigned
        to halos at zero redshift.

    Returns
    -------
    overlap : float
    """
    imax, jmax, kmax = delta1.shape

    totmass = 0.    # Total mass of clump 1 and clump 2
    intersect = 0.  # Mass of pixels that are non-zero in both clumps
    weight = 0.     # Weight to account for other halos
    count = 0       # Total number of pixels that are both non-zero

    for i in range(imax):
        ii = cellmins[0] + i
        for j in range(jmax):
            jj = cellmins[1] + j
            for k in range(kmax):
                kk = cellmins[2] + k

                cell1, cell2 = delta1[i, j, k], delta2[i, j, k]
                totmass += cell1 + cell2
                # If both are zero then skip
                if cell1 > 0 and cell2 > 0:
                    intersect += cell1 + cell2
                    weight += cell2 / delta2_full[ii, jj, kk]
                    count += 1

    # Normalise the intersect and weights
    intersect *= 0.5
    weight = weight / count if count > 0 else 0.
    return weight * intersect / (totmass - intersect)
