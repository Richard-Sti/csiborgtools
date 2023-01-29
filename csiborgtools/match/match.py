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

    def cross_knn_position_single(self, n_sim, nmult=5, dlogmass=None,
                                  mass_kind="totpartmass", overlap=False,
                                  overlapper_kwargs={}, select_initial=True,
                                  get_spherical_overlap=True,
                                  remove_nooverlap=True, verbose=True):
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
            Multiple of :math:`R_{\rm init}` or :math:`R_{200c}` within which
            to return neighbours. By default 5.
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
        get_spherical_overlap : bool, optional
            Whether to calculate the approximate spherical overlap in the
            initial snapshot. The radii are considered to be patch sizes.
            By default `True`.
        remove_nooverlap : bool, optional
            Whether to remove pairs with exactly zero overlap. By default
            `True`.
        verbose : bool, optional
            Iterator verbosity flag. By default `True`.

        Returns
        -------
        matches : composite array
            Array, indices are `(n_sims - 1, 5, n_halos, n_matches)`. The
            2nd axis is `index` of the neighbouring halo in its catalogue,
            `dist` is the 3D distance to the halo whose neighbours are
            searched, `dist0` is the separation of the initial CMs,
            `overlap` is the overlap over the initial clumps, and
            `spherical_overlap` is the approximate spherical overlap,
            all respectively.
        """
        self._check_masskind(mass_kind)
        # Halo properties of this simulation
        logmass = numpy.log10(self.cats[n_sim][mass_kind])
        pos = self.cats[n_sim].positions        # Grav potential minimum
        pos0 = self.cats[n_sim].positions0      # CM positions
        if select_initial:
            R = self.cats[n_sim]["patch_size"]  # Initial Lagrangian patch size
        else:
            R = self.cats[n_sim]["r200"]        # R200c at z = 0

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
            if select_initial:
                dist0, indxs = self.cats[i].radius_initial_neigbours(
                    pos0, R * nmult)
            else:
                # Will switch dist0 <-> dist at the end
                dist0, indxs = self.cats[i].radius_neigbours(
                    pos, R * nmult)
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

            # Calculate the approximate spherical overlap
            sphcross = [numpy.asanyarray([], dtype=numpy.float32)] * dist0.size
            if select_initial and get_spherical_overlap:
                for k in with_neigbours:
                    sphcrosses = numpy.full(indxs[k].size, numpy.nan,
                                            numpy.float32)
                    for ii, ind in enumerate(indxs[k]):
                        sphcrosses[ii] = spherical_overlap(
                            R[k], self.cats[i]["patch_size"][ind],
                            dist0[k][ii])

                    sphcross[k] = sphcrosses

            # Calculate the initial snapshot overlap
            cross = [numpy.asanyarray([], dtype=numpy.float32)] * dist0.size
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
                if verbose:
                    print("Smoothed up the field.", flush=True)
                # Convert positions to cell IDs only now after full delta
                clumps_pos2cell(clumpsx, overlapper)
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

                    # Unpack this clum and its mins and maxs
                    cl0 = clumps0["clump"][match0]
                    mins0_current, maxs0_current = mins0[match0], maxs0[match0]
                    # Preallocate this array.
                    crosses = numpy.full(indxs[k].size, numpy.nan,
                                         numpy.float32)

                    # Loop over the ones we cross-correlate with
                    for ii, ind in enumerate(indxs[k]):
                        # Again which cross clump to this index
                        matchx = cat2clumpsx[ind]
                        crosses[ii] = overlapper(
                            cl0, clumpsx["clump"][matchx], delta,
                            mins0_current, maxs0_current,
                            minsx[matchx], maxsx[matchx])

                    cross[k] = crosses
                    # Optionally remove points whose overlap is exaclt zero
                    if remove_nooverlap:
                        mask = cross[k] > 0
                        indxs[k] = indxs[k][mask]
                        dist[k] = dist[k][mask]
                        dist0[k] = dist0[k][mask]
                        cross[k] = cross[k][mask]
                        sphcross[k] = sphcross[k][mask]

            # Append as a composite array. Flip dist order if not select_init
            if select_initial:
                matches[count] = numpy.asarray(
                    [indxs, dist, dist0, cross, sphcross], dtype=object)
            else:
                matches[count] = numpy.asarray(
                    [indxs, dist0, dist, cross, sphcross], dtype=object)

        return numpy.asarray(matches, dtype=object)

    def cross_knn_position_all(self, nmult=5, dlogmass=None,
                               mass_kind="totpartmass", init_dist=False,
                               overlap=False, overlapper_kwargs={},
                               select_initial=True, get_spherical_overlap=True,
                               remove_nooverlap=True, verbose=True):
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
        get_spherical_overlap : bool, optional
            Whether to calculate the approximate spherical overlap in the
            initial snapshot. The radii are considered to be patch sizes.
            By default `True`.
        remove_nooverlap : bool, optional
            Whether to remove pairs with exactly zero overlap. By default
            `True`.
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
                i, nmult, dlogmass, mass_kind=mass_kind,
                init_dist=init_dist, overlap=overlap,
                overlapper_kwargs=overlapper_kwargs,
                select_initial=select_initial,
                get_spherical_overlap=get_spherical_overlap,
                remove_nooverlap=remove_nooverlap, verbose=verbose)
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
                    mins2=None, maxs2=None):
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

        # Preallocate and fill the array
        delta1 = numpy.zeros((ncells,)*3, dtype=numpy.float32)
        fill_delta(delta1, xc1, yc1, zc1, *cellmins, clump1['M'])
        delta2 = numpy.zeros((ncells,)*3, dtype=numpy.float32)
        fill_delta(delta2, xc2, yc2, zc2, *cellmins, clump2['M'])

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

    def __call__(self, clump1, clump2, delta2_full, mins1=None, maxs1=None,
                 mins2=None, maxs2=None):
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

        Returns
        -------
        overlap : float
        """
        delta1, delta2, cellmins = self.make_deltas(
            clump1, clump2, mins1, maxs1, mins2, maxs2)
        return _calculate_overlap(delta1, delta2, cellmins, delta2_full)


@jit(nopython=True)
def fill_delta(delta, xcell, ycell, zcell, xmin, ymin, zmin, weights):
    """
    Fill array delta at the specified indices with their weights. This is a JIT
    implementation.

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
    for i in range(xcell.size):
        delta[xcell[i] - xmin, ycell[i] - ymin, zcell[i] - zmin] += weights[i]


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

    i0, j0, k0 = cellmins  # Unpack things
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
def spherical_overlap(R1, R2, d):
    """
    Calculate the volume overlap between two spheres. Defined as their volume
    intersection divided by their total volume without the intersection.


    Parameters
    ----------
    R1, R2 : float
        Radius of the two spheres.
    d : float
        Separation of the spheres' centres.

    Returns
    -------
    overlap : float
    """
    # We want R1 >= R2 so optionally flip them.
    if R2 > R1:
        R1, R2 = R2, R1
    if d < R1 - R2:  # If S2 is entirely in S1
        return (R2 / R1)**3
    elif d > R1 + R2:  # If too far
        return 0.
    else:
        Vx = (R1 + R2 - d)**2 / (16 * d)
        Vx *= (d**2 + 2 * d * (R1 + R2) + 6 * R1 * R2 - 3 * (R1**2 + R2**2))
        return Vx / (R1**3 + R2**3 - Vx)
