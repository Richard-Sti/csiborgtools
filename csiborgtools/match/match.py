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
from gc import collect
from ..read import (CombinedHaloCatalogue, concatenate_clumps, clumps_pos2cell)  # noqa


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

    def cross_knn_position_single(self, n_sim, nmult=1, dlogmass=None,
                                  mass_kind="totpartmass", overlap=False,
                                  overlapper_kwargs={}, select_initial=True,
                                  remove_nooverlap=True, fast_neighbours=False,
                                  verbose=True):
        r"""
        Find all neighbours within a multiple of the sum of either the initial
        Lagrangian patch sizes (distance at :math:`z = 70`) or :math:`R_{200c}`
        (distance at :math:`z = 0`). Enforces that the neighbours' are similar
        in mass up to `dlogmass` dex and optionally calculates their overlap.

        Parameters
        ----------
        n_sim : int
            Index of an IC realisation in `self.cats` whose halos' neighbours
            in the remaining simulations to search for.
        nmult : float or int, optional
            Multiple of the sum of pair Lagrangian patch sizes or
            :math:`R_{200c}` within which to return neighbours. By default 1.
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
        remove_nooverlap : bool, optional
            Whether to remove pairs with exactly zero overlap. By default
            `True`.
        fast_neighbours : bool, optional
            Whether to calculate neighbours within a fixed radius of each
            clump. Note that this will result in missing some matches. If
            `True` then `nmult` is a multiple of either the initial patch size
            of :math:`R_{200c}`.
        verbose : bool, optional
            Iterator verbosity flag. By default `True`.

        Returns
        -------
        matches : composite array
            Array, indices are `(n_sims - 1, 5, n_halos, n_matches)`. The
            2nd axis is `index` of the neighbouring halo in its catalogue,
            `dist` is the 3D distance to the halo whose neighbours are
            searched, `dist0` is the separation of the initial CMs, and
            `overlap` is the overlap over the initial clumps, respectively.
        """
        self._check_masskind(mass_kind)
        # Halo properties of this simulation
        logmass = numpy.log10(self.cats[n_sim][mass_kind])
        pos = self.cats[n_sim].positions      # Grav potential minimum
        pos0 = self.cats[n_sim].positions0    # CM positions
        if select_initial:
            R = self.cats[n_sim]["lagpatch"]  # Initial Lagrangian patch size
        else:
            R = self.cats[n_sim]["r200"]      # R200c at z = 0

        if overlap:
            overlapper = ParticleOverlap(**overlapper_kwargs)
            if verbose:
                print("Loading initial clump particles for `n_sim = {}`."
                      .format(n_sim), flush=True)
            # Grab a paths object. What it is set to is unimportant
            paths = self.cats[0].paths
            with open(paths.clump0_path(self.cats.n_sims[n_sim]), "rb") as f:
                clumps0 = numpy.load(f, allow_pickle=True)
            clumps_pos2cell(clumps0, overlapper)
            # Precalculate min and max cell along each axis
            mins0, maxs0 = get_clumplims(clumps0,
                                         ncells=overlapper.inv_clength,
                                         nshift=overlapper.nshift)
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
            if verbose:
                print("Querying the KNN for `n_sim = {}`.".format(n_sim),
                      flush=True)
            # Query the KNN either fast (miss some) or slow (get all)
            if select_initial:
                if fast_neighbours:
                    dist0, indxs = self.cats[i].radius_neigbours(
                        pos0, R * nmult, select_initial=True)
                else:
                    dist0, indxs = radius_neighbours(
                        self.cats[i].knn0, pos0, radiusX=R,
                        radiusKNN=self.cats[i]["lagpatch"], nmult=nmult,
                        verbose=verbose)
            else:
                # Will switch dist0 <-> dist at the end
                if fast_neighbours:
                    dist0, indxs = self.cats[i].radius_neigbours(
                        pos, R * nmult, select_initial=False)
                else:
                    dist0, indxs = radius_neighbours(
                        self.cats[i].knn, pos, radiusX=R,
                        radiusKNN=self.cats[i]["r200"], nmult=nmult,
                        verbose=verbose)
            # Enforce int32 and float32
            for n in range(dist0.size):
                dist0[n] = dist0[n].astype(numpy.float32)
                indxs[n] = indxs[n].astype(numpy.int32)

            # Get rid of neighbors whose mass is too off
            if dlogmass is not None:
                for j, indx in enumerate(indxs):
                    match_logmass = numpy.log10(self.cats[i][mass_kind][indx])
                    mask = numpy.abs(match_logmass - logmass[j]) < dlogmass
                    dist0[j] = dist0[j][mask]
                    indxs[j] = indx[mask]

            # Find the distance at z = 0 (or z = 70 dep. on `search_initial``)
            dist = [numpy.asanyarray([], dtype=numpy.float32)] * dist0.size
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
            cross = [numpy.asanyarray([], dtype=numpy.float32)] * dist0.size
            if overlap:
                if verbose:
                    print("Loading initial clump particles for `n_sim = {}` "
                          "to compare against `n_sim = {}`.".format(i, n_sim),
                          flush=True)
                with open(paths.clump0_path(self.cats.n_sims[i]), 'rb') as f:
                    clumpsx = numpy.load(f, allow_pickle=True)
                clumps_pos2cell(clumpsx, overlapper)

                # Calculate the particle field
                if verbose:
                    print("Loading and smoothing the crossed total field.",
                          flush=True)
                particles = concatenate_clumps(clumpsx)
                delta = overlapper.make_delta(particles, to_smooth=False)
                del particles; collect()  # noqa - no longer needed
                delta = overlapper.smooth_highres(delta)
                if verbose:
                    print("Smoothed up the field.", flush=True)
                # Precalculate min and max cell along each axis
                minsx, maxsx = get_clumplims(clumpsx,
                                             ncells=overlapper.inv_clength,
                                             nshift=overlapper.nshift)

                cat2clumpsx = self._cat2clump_mapping(self.cats[i]["index"],
                                                      clumpsx["ID"])
                # Loop only over halos that have neighbours
                for k in tqdm(with_neigbours) if verbose else with_neigbours:
                    # Find which clump matches index of this halo from cat
                    match0 = cat2clumps0[k]

                    # Unpack this clum, its mamss and mins and maxs
                    cl0 = clumps0["clump"][match0]
                    mass0 = numpy.sum(cl0['M'])
                    mins0_current, maxs0_current = mins0[match0], maxs0[match0]
                    # Preallocate this array.
                    crosses = numpy.full(indxs[k].size, numpy.nan,
                                         numpy.float32)

                    # Loop over the ones we cross-correlate with
                    for ii, ind in enumerate(indxs[k]):
                        # Again which cross clump to this index
                        matchx = cat2clumpsx[ind]
                        clx = clumpsx["clump"][matchx]
                        crosses[ii] = overlapper(
                            cl0, clx, delta, mins0_current, maxs0_current,
                            minsx[matchx], maxsx[matchx],
                            mass1=mass0, mass2=numpy.sum(clx['M']))

                    cross[k] = crosses
                    # Optionally remove points whose overlap is exaclt zero
                    if remove_nooverlap:
                        mask = cross[k] > 0
                        indxs[k] = indxs[k][mask]
                        dist[k] = dist[k][mask]
                        dist0[k] = dist0[k][mask]
                        cross[k] = cross[k][mask]

            # Append as a composite array. Flip dist order if not select_init
            if select_initial:
                matches[count] = numpy.asarray(
                    [indxs, dist, dist0, cross], dtype=object)
            else:
                matches[count] = numpy.asarray(
                    [indxs, dist0, dist, cross], dtype=object)

        return numpy.asarray(matches, dtype=object)


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
        Convert position to cell number. If `pos` is in
        `numpy.typecodes["AllInteger"]` assumes it to already be the cell
        number.

        Parameters
        ----------
        pos : 1-dimensional array

        Returns
        -------
        cells : 1-dimensional array
        """
        # Check whether this is already the cell
        if pos.dtype.char in numpy.typecodes["AllInteger"]:
            return pos
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

    def make_delta(self, clump, mins=None, maxs=None, subbox=False,
                   to_smooth=True):
        """
        Calculate a NGP density field of a halo on a cubic grid.

        Parameters
        ----------
        clump: structurered arrays
            Clump structured array, keys must include `x`, `y`, `z` and `M`.
        mins, maxs : 1-dimensional arrays of shape `(3,)`
            Minimun and maximum cell numbers along each dimension.
        subbox : bool, optional
            Whether to calculate the density field on a grid strictly enclosing
            the clump.
        to_smooth : bool, optional
            Explicit control over whether to smooth. By default `True`.

        Returns
        -------
        delta : 3-dimensional array
        """
        cells = [self.pos2cell(clump[p]) for p in ('x', 'y', 'z')]

        # Check that minima and maxima are integers
        if not (mins is None and maxs is None):
            assert mins.dtype.char in numpy.typecodes["AllInteger"]
            assert maxs.dtype.char in numpy.typecodes["AllInteger"]

        if subbox:
            # Minimum xcell, ycell and zcell of this clump
            if mins is None or maxs is None:
                mins = numpy.asanyarray(
                    [max(numpy.min(cell) - self.nshift, 0) for cell in cells])
                maxs = numpy.asanyarray(
                    [min(numpy.max(cell) + self.nshift, self.inv_clength)
                     for cell in cells])

            ncells = numpy.max(maxs - mins) + 1  # To get the number of cells
        else:
            mins = (0, 0, 0,)
            ncells = self.inv_clength

        # Preallocate and fill the array
        delta = numpy.zeros((ncells,) * 3, dtype=numpy.float32)
        fill_delta(delta, *cells, *mins, clump['M'])

        if to_smooth and self.smooth_scale is not None:
            gaussian_filter(delta, self.smooth_scale, output=delta)
        return delta

    def make_deltas(self, clump1, clump2, mins1=None, maxs1=None,
                    mins2=None, maxs2=None, return_nonzero1=False):
        """
        Calculate a NGP density fields of two halos on a grid that encloses
        them both.

        Parameters
        ----------
        clump1, clump2 : structurered arrays
            Particle structured array of the two clumps. Keys must include `x`,
            `y`, `z` and `M`.
        mins1, maxs1 : 1-dimensional arrays of shape `(3,)`
            Minimun and maximum cell numbers along each dimension of `clump1`.
            Optional.
        mins2, maxs2 : 1-dimensional arrays of shape `(3,)`
            Minimun and maximum cell numbers along each dimension of `clump2`.
            Optional.

        Returns
        -------
        delta1, delta2 : 3-dimensional arrays
            Density arrays of `clump1` and `clump2`, respectively.
        cellmins : len-3 tuple
            Tuple of left-most cell ID in the full box.
        nonzero1 : 2-dimensional array
            Indices where `delta1` has a non-zero density. If `return_nonzero1`
            is `False` return `None` instead.
        """
        xc1, yc1, zc1 = (self.pos2cell(clump1[p]) for p in ('x', 'y', 'z'))
        xc2, yc2, zc2 = (self.pos2cell(clump2[p]) for p in ('x', 'y', 'z'))

        if any(obj is None for obj in (mins1, maxs1, mins2, maxs2)):
            # Minimum cell number of the two halos along each dimension
            xmin = min(numpy.min(xc1), numpy.min(xc2)) - self.nshift
            ymin = min(numpy.min(yc1), numpy.min(yc2)) - self.nshift
            zmin = min(numpy.min(zc1), numpy.min(zc2)) - self.nshift
            # Make sure shifting does not go beyond boundaries
            xmin, ymin, zmin = [max(px, 0) for px in (xmin, ymin, zmin)]

            # Maximum cell number of the two halos along each dimension
            xmax = max(numpy.max(xc1), numpy.max(xc2)) + self.nshift
            ymax = max(numpy.max(yc1), numpy.max(yc2)) + self.nshift
            zmax = max(numpy.max(zc1), numpy.max(zc2)) + self.nshift
            # Make sure shifting does not go beyond boundaries
            xmax, ymax, zmax = [min(px, self.inv_clength - 1)
                                for px in (xmax, ymax, zmax)]
        else:
            xmin, ymin, zmin = [min(mins1[i], mins2[i]) for i in range(3)]
            xmax, ymax, zmax = [max(maxs1[i], maxs2[i]) for i in range(3)]

        cellmins = (xmin, ymin, zmin, )  # Cell minima
        ncells = max(xmax - xmin, ymax - ymin, zmax - zmin) + 1  # Num cells

        # Preallocate and fill the arrays
        delta1 = numpy.zeros((ncells,)*3, dtype=numpy.float32)
        delta2 = numpy.zeros((ncells,)*3, dtype=numpy.float32)
        if return_nonzero1:
            nonzero1 = fill_delta_indxs(
                delta1, xc1, yc1, zc1, *cellmins, clump1['M'])
        else:
            fill_delta(delta1, xc1, yc1, zc1, *cellmins, clump1['M'])
            nonzero1 = None
        fill_delta(delta2, xc2, yc2, zc2, *cellmins, clump2['M'])

        if self.smooth_scale is not None:
            gaussian_filter(delta1, self.smooth_scale, output=delta1)
            gaussian_filter(delta2, self.smooth_scale, output=delta2)

        return delta1, delta2, cellmins, nonzero1

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
        return calculate_overlap(delta1, delta2, cellmins, delta2_full)

    def __call__(self, clump1, clump2, delta2_full, mins1=None, maxs1=None,
                 mins2=None, maxs2=None, mass1=None, mass2=None,
                 loop_nonzero=True):
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
        mins1, maxs1 : 1-dimensional arrays of shape `(3,)`
            Minimun and maximum cell numbers along each dimension of `clump1`.
            Optional.
        mins2, maxs2 : 1-dimensional arrays of shape `(3,)`
            Minimun and maximum cell numbers along each dimension of `clump2`.
            Optional.
        mass1, mass2 : floats, optional
            Total mass of `clump1` and `clump2`, respectively. Must be provided
            if `loop_nonzero` is `True`.
        loop_nonzer : bool, optional
            Whether to only loop over cells where `clump1` has non-zero
            density. By default `True`.

        Returns
        -------
        overlap : float
        """
        delta1, delta2, cellmins, nonzero1 = self.make_deltas(
            clump1, clump2, mins1, maxs1, mins2, maxs2,
            return_nonzero1=loop_nonzero)

        if not loop_nonzero:
            return calculate_overlap(delta1, delta2, cellmins, delta2_full)

        return calculate_overlap_indxs(delta1, delta2, cellmins, delta2_full,
                                       nonzero1, mass1, mass2)


@jit(nopython=True)
def fill_delta(delta, xcell, ycell, zcell, xmin, ymin, zmin, weights):
    """
    Fill array `delta` at the specified indices with their weights. This is a
    JIT implementation.

    Parameters
    ----------
    delta : 3-dimensional array
        Grid to be filled with weights.
    xcell, ycell, zcell : 1-dimensional arrays
        Indices where to assign `weights`.
    xmin, ymin, zmin : ints
        Minimum cell IDs of particles.
    weights : 1-dimensional arrays
        Particle mass.

    Returns
    -------
    None
    """
    for n in range(xcell.size):
        delta[xcell[n] - xmin, ycell[n] - ymin, zcell[n] - zmin] += weights[n]


@jit(nopython=True)
def fill_delta_indxs(delta, xcell, ycell, zcell, xmin, ymin, zmin, weights):
    """
    Fill array `delta` at the specified indices with their weights and return
    indices where `delta` was assigned a value. This is a JIT implementation.

    Parameters
    ----------
    delta : 3-dimensional array
        Grid to be filled with weights.
    xcell, ycell, zcell : 1-dimensional arrays
        Indices where to assign `weights`.
    xmin, ymin, zmin : ints
        Minimum cell IDs of particles.
    weights : 1-dimensional arrays
        Particle mass.

    Returns
    -------
    cells : 1-dimensional array
        Indices where `delta` was assigned a value.
    """
    # Array to count non-zero cells
    cells = numpy.full((xcell.size, 3), numpy.nan, numpy.int32)
    count_nonzero = 0
    for n in range(xcell.size):
        i, j, k = xcell[n] - xmin, ycell[n] - ymin, zcell[n] - zmin
        # If a cell is zero add it
        if delta[i, j, k] == 0:
            cells[count_nonzero, :] = i, j, k
            count_nonzero += 1

        delta[i, j, k] += weights[n]

    return cells[:count_nonzero, :]  # Cutoff unassigned places


def get_clumplims(clumps, ncells, nshift=None):
    """
    Get the lower and upper limit of clumps' positions or cell numbers.

    Parameters
    ----------
    clumps : array of arrays
        Array of clump structured arrays.
    ncells : int
        Number of grid cells of the box along a single dimension.
    nshift : int, optional
        Lower and upper shift of the clump limits.

    Returns
    -------
    mins, maxs : 2-dimensional arrays of shape `(n_samples, 3)`
        The minimum and maximum along each axis.
    """
    dtype = clumps[0][0]['x'].dtype  # dtype of the first clump's 'x'
    # Check that for real positions we cannot apply nshift
    if nshift is not None and dtype.char not in numpy.typecodes["AllInteger"]:
        raise TypeError("`nshift` supported only positions are cells.")
    nshift = 0 if nshift is None else nshift  # To simplify code below

    nclumps = clumps.size
    mins = numpy.full((nclumps, 3), numpy.nan, dtype=dtype)
    maxs = numpy.full((nclumps, 3), numpy.nan, dtype=dtype)

    for i, clump in enumerate(clumps):
        for j, p in enumerate(['x', 'y', 'z']):
            mins[i, j] = max(numpy.min(clump[0][p]) - nshift, 0)
            maxs[i, j] = min(numpy.max(clump[0][p]) + nshift, ncells - 1)

    return mins, maxs


@jit(nopython=True)
def calculate_overlap(delta1, delta2, cellmins, delta2_full):
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
    totmass = 0.           # Total mass of clump 1 and clump 2
    intersect = 0.         # Mass of pixels that are non-zero in both clumps
    weight = 0.            # Weight to account for other halos
    count = 0              # Total number of pixels that are both non-zero
    i0, j0, k0 = cellmins  # Unpack things
    imax, jmax, kmax = delta1.shape

    for i in range(imax):
        ii = i0 + i
        for j in range(jmax):
            jj = j0 + j
            for k in range(kmax):
                kk = k0 + k

                cell1, cell2 = delta1[i, j, k], delta2[i, j, k]
                cell = cell1 + cell2
                totmass += cell
                # If both are zero then skip
                if cell1 * cell2 > 0:
                    intersect += cell
                    weight += cell2 / delta2_full[ii, jj, kk]
                    count += 1

    # Normalise the intersect and weights
    intersect *= 0.5
    weight = weight / count if count > 0 else 0.
    return weight * intersect / (totmass - intersect)


@jit(nopython=True)
def calculate_overlap_indxs(delta1, delta2, cellmins, delta2_full, nonzero1,
                            mass1, mass2):
    r"""
    Overlap between two clumps whose density fields are evaluated on the
    same grid and `nonzero1` enumerates the non-zero cells of `delta1.  This is
    a JIT implementation, hence it is outside of the main class.

    Parameters
    ----------
    delta1, delta2 : 3-dimensional arrays
        Clumps density fields.
    cellmins : len-3 tuple
        Tuple of left-most cell ID in the full box.
    delta2_full : 3-dimensional array
        Density field of the whole box calculated with particles assigned
        to halos at zero redshift.
    nonzero1 : 2-dimensional array of shape `(n_cells, 3)`
        Indices of cells that are non-zero in `delta1`. Expected to be
        precomputed from `fill_delta_indxs`.
    mass1, mass2 : floats, optional
        Total masses of the two clumps, respectively. Optional. If not provided
        calculcated directly from the density field.

    Returns
    -------
    overlap : float
    """
    totmass = mass1 + mass2  # Total mass of clump 1 and clump 2
    intersect = 0.           # Mass of pixels that are non-zero in both clumps
    weight = 0.              # Weight to account for other halos
    count = 0                # Total number of pixels that are both non-zero
    i0, j0, k0 = cellmins    # Unpack cell minimas

    ncells = nonzero1.shape[0]

    for n in range(ncells):
        i, j, k = nonzero1[n, :]
        cell1, cell2 = delta1[i, j, k], delta2[i, j, k]

        if cell2 > 0:  # We already know that cell1 is non-zero
            intersect += cell1 + cell2
            weight += cell2 / delta2_full[i0 + i, j0 + j, k0 + k]
            count += 1

    # Normalise the intersect and weights
    intersect *= 0.5
    weight = weight / count if count > 0 else 0.
    return weight * intersect / (totmass - intersect)


def dist_centmass(clump):
    """
    Calculate the clump particles' distance from the centre of mass.

    Parameters
    ----------
    clump : structurered arrays
        Clump structured array. Keyes must include `x`, `y`, `z` and `M`.

    Returns
    -------
    dist : 1-dimensional array of shape `(n_particles, )`
        Particle distance from the centre of mass.
    cm : 1-dimensional array of shape `(3,)`
        Center of mass coordinates.
    """
    # CM along each dimension
    cmx, cmy, cmz = [numpy.average(clump[p], weights=clump['M'])
                     for p in ('x', 'y', 'z')]
    # Particle distance from the CM
    dist = numpy.sqrt(numpy.square(clump['x'] - cmx)
                      + numpy.square(clump['y'] - cmy)
                      + numpy.square(clump['z'] - cmz))

    return dist, numpy.asarray([cmx, cmy, cmz])


def dist_percentile(dist, qs, distmax=0.075):
    """
    Calculate q-th percentiles of `dist`, with an upper limit of `distmax`.

    Parameters
    ----------
    dist : 1-dimensional array
        Array of distances.
    qs : 1-dimensional array
        Percentiles to compute.
    distmax : float, optional
        The maximum distance. By default 0.075.

    Returns
    -------
    x : 1-dimensional array
    """
    x = numpy.percentile(dist, qs)
    x[x > distmax] = distmax  # Enforce the upper limit
    return x


def radius_neighbours(knn, X, radiusX, radiusKNN, nmult=1., verbose=True):
    """
    Find all neigbours of a trained KNN model whose center of mass separation
    is less than `nmult` times the sum of their respective radii.

    Parameters
    ----------
    knn : :py:class:`sklearn.neighbors.NearestNeighbors`
        Fitted nearest neighbour search.
    X : 2-dimensional array
        Array of shape `(n_samples, 3)`, where the latter axis represents
        `x`, `y` and `z`.
    radiusX: 1-dimensional array of shape `(n_samples, )`
        Patch radii corresponding to clumps in `X`.
    radiusKNN : 1-dimensional array
        Patch radii corresponding to clumps used to train `knn`.
    nmult : float, optional
        Multiple of the sum of two radii below which to consider a match.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    dists : 1-dimensional array `(n_samples,)` of arrays
        Distance from `X` to matches from `knn`.
    indxs : 1-dimensional array `(n_samples,)` of arrays
        Matches to `X` from `knn`.
    """
    assert X.ndim == 2 and X.shape[1] == 3       # shape of X ok?
    assert X.shape[0] == radiusX.size            # patchX matches X?
    assert radiusKNN.size == knn.n_samples_fit_  # patchknn matches the knn?

    nsamples = X.shape[0]
    dists = [None] * nsamples            # Initiate lists
    indxs = [None] * nsamples
    patchknn_max = numpy.max(radiusKNN)  # Maximum for completeness

    for i in trange(nsamples) if verbose else range(nsamples):
        dist, indx = knn.radius_neighbors(X[i, :].reshape(-1, 3),
                                          radiusX[i] + patchknn_max,
                                          sort_results=True)
        # Note that `dist` and `indx` are wrapped in 1-element arrays
        # so we take the first item where appropriate
        mask = (dist[0] / (radiusX[i] + radiusKNN[indx[0]])) < nmult
        dists[i] = dist[0][mask]
        indxs[i] = indx[0][mask]

    dists = numpy.asarray(dists, dtype=object)  # Turn into array of arrays
    indxs = numpy.asarray(indxs, dtype=object)
    return dists, indxs
