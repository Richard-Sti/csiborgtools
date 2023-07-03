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
Support for matching halos between CSiBORG IC realisations.
"""
from datetime import datetime
from functools import lru_cache
from math import ceil

import numpy
from numba import jit
from scipy.ndimage import gaussian_filter
from tqdm import tqdm, trange

from ..read import load_parent_particles

BCKG_HALFSIZE = 475
BOX_SIZE = 2048

###############################################################################
#                  Realisations matcher for calculating overlaps              #
###############################################################################


class RealisationsMatcher:
    """
    A tool to match halos between IC realisations.

    Parameters
    ----------
    nmult : float or int, optional
        Multiple of the sum of pair initial Lagrangian patch sizes
        within which to return neighbours. By default 1.
    dlogmass : float, optional
        Tolerance on the absolute logarithmic mass difference of potential
        matches. By default 2.
    mass_kind : str, optional
        The mass kind whose similarity is to be checked. Must be a valid
        catalogue key. By default `totpartmass`, i.e. the total particle
        mass associated with a halo.
    """

    _nmult = None
    _dlogmass = None
    _mass_kind = None
    _overlapper = None

    def __init__(self, nmult=1.0, dlogmass=2.0, mass_kind="totpartmass"):
        assert nmult > 0
        assert dlogmass > 0
        assert isinstance(mass_kind, str)
        self._nmult = nmult
        self._dlogmass = dlogmass
        self._mass_kind = mass_kind
        self._overlapper = ParticleOverlap()

    @property
    def nmult(self):
        """
        Multiple of the sum of pair initial Lagrangian patch sizes within which
        to return neighbours.

        Returns
        -------
        nmult : float
        """
        return self._nmult

    @property
    def dlogmass(self):
        """
        Tolerance on the absolute logarithmic mass difference of potential
        matches.

        Returns
        -------
        dlogmass : float
        """
        return self._dlogmass

    @property
    def mass_kind(self):
        """
        Mass kind whose similarity is to be checked.

        Returns
        -------
        mass_kind : str
        """
        return self._mass_kind

    @property
    def overlapper(self):
        """
        The overlapper object.

        Returns
        -------
        overlapper : :py:class:`csiborgtools.match.ParticleOverlap`
        """
        return self._overlapper

    def cross(self, cat0, catx, particles0, particlesx, clump_map0, clump_mapx,
              delta_bckg, cache_size=10000, verbose=True):
        r"""
        Find all neighbours whose CM separation is less than `nmult` times the
        sum of their initial Lagrangian patch sizes and calculate their
        overlap. Enforces that the neighbours' are similar in mass up to
        `dlogmass` dex.

        Parameters
        ----------
        cat0 : :py:class:`csiborgtools.read.CSiBORGHaloCatalogue`
            Halo catalogue of the reference simulation.
        catx : :py:class:`csiborgtools.read.CSiBORGHaloCatalogue`
            Halo catalogue of the cross simulation.
        particles0 : 2-dimensional array
            Array of particles in box units in the reference simulation.
            The columns must be `x`, `y`, `z` and `M`.
        particlesx : 2-dimensional array
            Array of particles in box units in the cross simulation.
            The columns must be `x`, `y`, `z` and `M`.
        clump_map0 : 2-dimensional array
            Clump map of the reference simulation.
        clump_mapx : 2-dimensional array
            Clump map of the cross simulation.
        delta_bckg : 3-dimensional array
            Summed background density field of the reference and cross
            simulations calculated with particles assigned to halos at the
            final snapshot. Assumed to only be sampled in cells
            :math:`[512, 1536)^3`.
        cache_size : int, optional
            Caching size for loading the cross simulation halos.
        verbose : bool, optional
            iterator verbosity flag. by default `true`.

        Returns
        -------
        match_indxs : 1-dimensional array of arrays
            The outer array corresponds to halos in the reference catalogue,
            the inner array corresponds to the array positions of matches in
            the cross catalogue.
        overlaps : 1-dimensional array of arrays
            Overlaps with the cross catalogue. Follows similar pattern as
            `match_indxs`.
        """
        # We begin by querying the kNN for the nearest neighbours of each halo
        # in the reference simulation from the cross simulation in the initial
        # snapshot.
        if verbose:
            print(f"{datetime.now()}: querying the KNN.", flush=True)
        match_indxs = radius_neighbours(
            catx.knn(in_initial=True), cat0.position(in_initial=True),
            radiusX=cat0["lagpatch_size"], radiusKNN=catx["lagpatch_size"],
            nmult=self.nmult, enforce_int32=True, verbose=verbose)

        # We next remove neighbours whose mass is too large/small.
        if self.dlogmass is not None:
            for i, indx in enumerate(match_indxs):
                # |log(M1 / M2)|
                p = self.mass_kind
                aratio = numpy.abs(numpy.log10(catx[p][indx] / cat0[p][i]))
                match_indxs[i] = match_indxs[i][aratio < self.dlogmass]

        clid2map0 = {clid: i for i, clid in enumerate(clump_map0[:, 0])}
        clid2mapx = {clid: i for i, clid in enumerate(clump_mapx[:, 0])}

        # We will cache the halos from the cross simulation to speed up the I/O
        @lru_cache(maxsize=cache_size)
        def load_cached_halox(hid):
            return load_processed_halo(hid, particlesx, clump_mapx, clid2mapx,
                                       catx.clumps_cat, nshift=0,
                                       ncells=BOX_SIZE)

        if verbose:
            print(f"{datetime.now()}: calculating overlaps.", flush=True)
        cross = [numpy.asanyarray([], dtype=numpy.float32)] * match_indxs.size
        indxs = cat0["index"]
        for i, k0 in enumerate(tqdm(indxs) if verbose else indxs):
            # If we have no matches continue to the next halo.
            matches = match_indxs[i]
            if matches.size == 0:
                continue
            # Next, we find this halo's particles, total mass, minimum and
            # maximum cells and convert positions to cells.
            pos0, mass0, totmass0, mins0, maxs0 = load_processed_halo(
                k0, particles0, clump_map0, clid2map0, cat0.clumps_cat,
                nshift=0, ncells=BOX_SIZE)

            # We now loop over matches of this halo and calculate their
            # overlap, storing them in `_cross`.
            _cross = numpy.full(matches.size, numpy.nan, dtype=numpy.float32)
            for j, kx in enumerate(catx["index"][matches]):
                posx, massx, totmassx, minsx, maxsx = load_cached_halox(kx)
                _cross[j] = self.overlapper(
                    pos0, posx, mass0, massx, delta_bckg, mins0, maxs0,
                    minsx, maxsx, totmass1=totmass0, totmass2=totmassx)
            cross[i] = _cross

            # We remove all matches that have zero overlap to save space.
            mask = cross[i] > 0
            match_indxs[i] = match_indxs[i][mask]
            cross[i] = cross[i][mask]
            # And finally we sort the matches by their overlap.
            ordering = numpy.argsort(cross[i])[::-1]
            match_indxs[i] = match_indxs[i][ordering]
            cross[i] = cross[i][ordering]

        cross = numpy.asanyarray(cross, dtype=object)
        return match_indxs, cross

    def smoothed_cross(self, cat0, catx, particles0, particlesx, clump_map0,
                       clump_mapx, delta_bckg, match_indxs, smooth_kwargs,
                       cache_size=10000, verbose=True):
        r"""
        Calculate the smoothed overlaps for pair previously identified via
        `self.cross(...)` to have a non-zero overlap.

        Parameters
        ----------
        cat0 : :py:class:`csiborgtools.read.ClumpsCatalogue`
            Halo catalogue of the reference simulation.
        catx : :py:class:`csiborgtools.read.ClumpsCatalogue`
            Halo catalogue of the cross simulation.
        particles0 : 2-dimensional array
            Array of particles in box units in the reference simulation.
            The columns must be `x`, `y`, `z` and `M`.
        particlesx : 2-dimensional array
            Array of particles in box units in the cross simulation.
            The columns must be `x`, `y`, `z` and `M`.
        clump_map0 : 2-dimensional array
            Clump map of the reference simulation.
        clump_mapx : 2-dimensional array
            Clump map of the cross simulation.
        delta_bckg : 3-dimensional array
            Smoothed summed background density field of the reference and cross
            simulations calculated with particles assigned to halos at the
            final snapshot. Assumed to only be sampled in cells
            :math:`[512, 1536)^3`.
        match_indxs : 1-dimensional array of arrays
            Indices of halo counterparts in the cross catalogue.
        smooth_kwargs : kwargs
            Kwargs to be passed to :py:func:`scipy.ndimage.gaussian_filter`.
        cache_size : int, optional
            Caching size for loading the cross simulation halos.
        verbose : bool, optional
            Iterator verbosity flag. By default `True`.

        Returns
        -------
        overlaps : 1-dimensional array of arrays
        """
        nshift = read_nshift(smooth_kwargs)
        clid2map0 = {clid: i for i, clid in enumerate(clump_map0[:, 0])}
        clid2mapx = {clid: i for i, clid in enumerate(clump_mapx[:, 0])}

        @lru_cache(maxsize=cache_size)
        def load_cached_halox(hid):
            return load_processed_halo(hid, particlesx, clump_mapx, clid2mapx,
                                       catx.clumps_cat, nshift=nshift,
                                       ncells=BOX_SIZE)

        if verbose:
            print(f"{datetime.now()}: calculating smoothed overlaps.",
                  flush=True)
        indxs = cat0["index"]
        cross = [numpy.asanyarray([], dtype=numpy.float32)] * match_indxs.size
        for i, k0 in enumerate(tqdm(indxs) if verbose else indxs):
            pos0, mass0, __, mins0, maxs0 = load_processed_halo(
                k0, particles0, clump_map0, clid2map0, cat0.clumps_cat,
                nshift=nshift, ncells=BOX_SIZE)

            # Now loop over the matches and calculate the smoothed overlap.
            _cross = numpy.full(match_indxs[i].size, numpy.nan, numpy.float32)
            for j, kx in enumerate(catx["index"][match_indxs[i]]):
                posx, massx, __, minsx, maxsx = load_cached_halox(kx)
                _cross[j] = self.overlapper(pos0, posx, mass0, massx,
                                            delta_bckg, mins0, maxs0, minsx,
                                            maxsx, smooth_kwargs=smooth_kwargs)
            cross[i] = _cross

        return numpy.asanyarray(cross, dtype=object)


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
    Class to calculate halo overlaps. The density field calculation is based on
    the nearest grid position particle assignment scheme, with optional
    Gaussian smoothing.
    """

    def make_bckg_delta(self, particles, clump_map, clid2map, halo_cat,
                        delta=None, verbose=False):
        """
        Calculate a NGP density field of particles belonging to halos of a
        halo catalogue `halo_cat`. Particles are only counted within the
        high-resolution region of the simulation. Smoothing must be applied
        separately.

        Parameters
        ----------
        particles : 2-dimensional array
            Array of particles.
        clump_map : 2-dimensional array
            Array containing start and end indices in the particle array
            corresponding to each clump.
        clid2map : dict
            Dictionary mapping clump IDs to `clump_map` array positions.
        halo_cat: :py:class:`csiborgtools.read.CSiBORGHaloCatalogue`
            Halo catalogue.
        delta : 3-dimensional array, optional
            Array to store the density field in. If `None` a new array is
            created.
        verbose : bool, optional
            Verbosity flag for loading the halos' particles.

        Returns
        -------
        delta : 3-dimensional array
        """
        cellmin = BOX_SIZE // 2 - BCKG_HALFSIZE
        cellmax = BOX_SIZE // 2 + BCKG_HALFSIZE
        ncells = cellmax - cellmin
        # We then pre-allocate the density field/check it is of the right shape
        if delta is None:
            delta = numpy.zeros((ncells,) * 3, dtype=numpy.float32)
        else:
            assert ((delta.shape == (ncells,) * 3)
                    & (delta.dtype == numpy.float32))

        clumps_cat = halo_cat.clumps_cat
        for hid in tqdm(halo_cat["index"]) if verbose else halo_cat["index"]:
            pos = load_parent_particles(hid, particles, clump_map, clid2map,
                                        clumps_cat)
            if pos is None:
                continue

            pos, mass = pos[:, :3], pos[:, 3]
            pos = pos2cell(pos, BOX_SIZE)
            # We mask out particles outside the cubical high-resolution region
            mask = numpy.all((cellmin <= pos) & (pos < cellmax), axis=1)
            pos = pos[mask]
            fill_delta(delta, pos[:, 0], pos[:, 1], pos[:, 2],
                       *(cellmin,) * 3, mass[mask])
        return delta

    def make_delta(self, pos, mass, mins=None, maxs=None, subbox=False,
                   smooth_kwargs=None):
        """
        Calculate a NGP density field of a halo on a cubic grid. Optionally can
        be smoothed with a Gaussian kernel.

        Parameters
        ----------
        pos : 2-dimensional array
            Halo particle position array.
        mass : 1-dimensional array
            Halo particle mass array.
        mins, maxs : 1-dimensional arrays of shape `(3,)`
            Minimun and maximum cell numbers along each dimension.
        subbox : bool, optional
            Whether to calculate the density field on a grid strictly enclosing
            the clump.
        smooth_kwargs : kwargs, optional
            Kwargs to be passed to :py:func:`scipy.ndimage.gaussian_filter`.
            If `None` no smoothing is applied.

        Returns
        -------
        delta : 3-dimensional array
        """
        nshift = read_nshift(smooth_kwargs)
        cells = pos2cell(pos, BOX_SIZE)
        # Check that minima and maxima are integers
        if not (mins is None and maxs is None):
            assert mins.dtype.char in numpy.typecodes["AllInteger"]
            assert maxs.dtype.char in numpy.typecodes["AllInteger"]

        if subbox:
            if mins is None or maxs is None:
                mins, maxs = get_halolims(cells, BOX_SIZE, nshift)

            ncells = maxs - mins + 1  # To get the number of cells
        else:
            mins = [0, 0, 0]
            ncells = (BOX_SIZE, ) * 3

        # Preallocate and fill the array
        delta = numpy.zeros(ncells, dtype=numpy.float32)
        fill_delta(delta, cells[:, 0], cells[:, 1], cells[:, 2], *mins, mass)
        if smooth_kwargs is not None:
            gaussian_filter(delta, output=delta, **smooth_kwargs)
        return delta

    def make_deltas(self, pos1, pos2, mass1, mass2, mins1=None, maxs1=None,
                    mins2=None, maxs2=None, smooth_kwargs=None):
        """
        Calculate a NGP density fields of two halos on a grid that encloses
        them both. Optionally can be smoothed with a Gaussian kernel.

        Parameters
        ----------
        pos1 : 2-dimensional array
            Particle positions of the first halo.
        pos2 : 2-dimensional array
            Particle positions of the second halo.
        mass1 : 1-dimensional array
            Particle masses of the first halo.
        mass2 : 1-dimensional array
            Particle masses of the second halo.
        mins1, maxs1 : 1-dimensional arrays of shape `(3,)`
            Minimun and maximum cell numbers along each dimension of `clump1`.
            Optional.
        mins2, maxs2 : 1-dimensional arrays of shape `(3,)`
            Minimun and maximum cell numbers along each dimension of `clump2`.
            Optional.
        smooth_kwargs : kwargs, optional
            Kwargs to be passed to :py:func:`scipy.ndimage.gaussian_filter`.
            If `None` no smoothing is applied.

        Returns
        -------
        delta1, delta2 : 3-dimensional arrays
            Density arrays of `clump1` and `clump2`, respectively.
        cellmins : len-3 tuple
            Tuple of left-most cell ID in the full box.
        nonzero : 2-dimensional array
            Indices where the lower mass clump has a non-zero density.
            Calculated only if no smoothing is applied, otherwise `None`.
        """
        nshift = read_nshift(smooth_kwargs)
        pos1 = pos2cell(pos1, BOX_SIZE)
        pos2 = pos2cell(pos2, BOX_SIZE)
        xc1, yc1, zc1 = [pos1[:, i] for i in range(3)]
        xc2, yc2, zc2 = [pos2[:, i] for i in range(3)]

        if any(obj is None for obj in (mins1, maxs1, mins2, maxs2)):
            # Minimum cell number of the two halos along each dimension
            xmin = min(numpy.min(xc1), numpy.min(xc2)) - nshift
            ymin = min(numpy.min(yc1), numpy.min(yc2)) - nshift
            zmin = min(numpy.min(zc1), numpy.min(zc2)) - nshift
            # Make sure shifting does not go beyond boundaries
            xmin, ymin, zmin = [max(px, 0) for px in (xmin, ymin, zmin)]

            # Maximum cell number of the two halos along each dimension
            xmax = max(numpy.max(xc1), numpy.max(xc2)) + nshift
            ymax = max(numpy.max(yc1), numpy.max(yc2)) + nshift
            zmax = max(numpy.max(zc1), numpy.max(zc2)) + nshift
            # Make sure shifting does not go beyond boundaries
            xmax, ymax, zmax = [min(px, BOX_SIZE - 1)
                                for px in (xmax, ymax, zmax)]
        else:
            xmin, ymin, zmin = [min(mins1[i], mins2[i]) for i in range(3)]
            xmax, ymax, zmax = [max(maxs1[i], maxs2[i]) for i in range(3)]

        cellmins = (xmin, ymin, zmin)  # Cell minima
        ncells = xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1  # Num cells

        # Preallocate and fill the arrays
        delta1 = numpy.zeros(ncells, dtype=numpy.float32)
        delta2 = numpy.zeros(ncells, dtype=numpy.float32)

        # If no smoothing figure out the nonzero indices of the smaller clump
        if smooth_kwargs is None:
            if pos1.shape[0] > pos2.shape[0]:
                fill_delta(delta1, xc1, yc1, zc1, *cellmins, mass1)
                nonzero = fill_delta_indxs(delta2, xc2, yc2, zc2, *cellmins,
                                           mass2)
            else:
                nonzero = fill_delta_indxs(delta1, xc1, yc1, zc1, *cellmins,
                                           mass1)
                fill_delta(delta2, xc2, yc2, zc2, *cellmins, mass2)
        else:
            fill_delta(delta1, xc1, yc1, zc1, *cellmins, mass1)
            fill_delta(delta2, xc2, yc2, zc2, *cellmins, mass2)
            nonzero = None

        if smooth_kwargs is not None:
            gaussian_filter(delta1, output=delta1, **smooth_kwargs)
            gaussian_filter(delta2, output=delta2, **smooth_kwargs)
        return delta1, delta2, cellmins, nonzero

    def __call__(self, pos1, pos2, mass1, mass2, delta_bckg,
                 mins1=None, maxs1=None, mins2=None, maxs2=None,
                 totmass1=None, totmass2=None, smooth_kwargs=None):
        """
        Calculate overlap between `clump1` and `clump2`. See
        `calculate_overlap(...)` for further information. Be careful so that
        the background density field is calculated with the same
        `smooth_kwargs`. If any smoothing is applied then loops over the full
        density fields, otherwise only over the non-zero cells of the lower
        mass clump.

        Parameters
        ----------
        pos1 : 2-dimensional array
            Particle positions of the first halo.
        pos2 : 2-dimensional array
            Particle positions of the second halo.
        mass1 : 1-dimensional array
            Particle masses of the first halo.
        mass2 : 1-dimensional array
            Particle masses of the second halo.
        cellmins : len-3 tuple
            Tuple of left-most cell ID in the full box.
        delta_bckg : 3-dimensional array
            Summed background density field of the reference and cross
            simulations calculated with particles assigned to halos at the
            final snapshot. Assumed to only be sampled in cells
            :math:`[512, 1536)^3`.
        mins1, maxs1 : 1-dimensional arrays of shape `(3,)`
            Minimun and maximum cell numbers along each dimension of `clump1`.
            Optional.
        mins2, maxs2 : 1-dimensional arrays of shape `(3,)`
            Minimum and maximum cell numbers along each dimension of `clump2`,
            optional.
        totmass1, totmass2 : floats, optional
            Total mass of `clump1` and `clump2`, respectively. Must be provided
            if `loop_nonzero` is `True`.
        smooth_kwargs : kwargs, optional
            Kwargs to be passed to :py:func:`scipy.ndimage.gaussian_filter`.
            If `None` no smoothing is applied.

        Returns
        -------
        overlap : float
        """
        delta1, delta2, cellmins, nonzero = self.make_deltas(
            pos1, pos2, mass1, mass2, mins1, maxs1, mins2, maxs2,
            smooth_kwargs=smooth_kwargs)

        if smooth_kwargs is not None:
            return calculate_overlap(delta1, delta2, cellmins, delta_bckg)
        # Calculate masses not given
        totmass1 = numpy.sum(mass1) if totmass1 is None else totmass1
        totmass2 = numpy.sum(mass2) if totmass2 is None else totmass2
        return calculate_overlap_indxs(
            delta1, delta2, cellmins, delta_bckg, nonzero, totmass1, totmass2)


###############################################################################
#                     Halo matching supplementary functions                   #
###############################################################################


def pos2cell(pos, ncells):
    """
    Convert position to cell number. If `pos` is in
    `numpy.typecodes["AllInteger"]` assumes it to already be the cell
    number.

    Parameters
    ----------
    pos : 1-dimensional array
        Array of positions along an axis in the box.
    ncells : int
        Number of cells along the axis.

    Returns
    -------
    cells : 1-dimensional array
    """
    if pos.dtype.char in numpy.typecodes["AllInteger"]:
        return pos
    return numpy.floor(pos * ncells).astype(numpy.int32)


def read_nshift(smooth_kwargs):
    """
    Read off the number of cells to pad the density field if smoothing is
    applied. Defaults to the ceiling of twice of the smoothing scale.

    Parameters
    ----------
    smooth_kwargs : kwargs, optional
        Kwargs to be passed to :py:func:`scipy.ndimage.gaussian_filter`.
        If `None` no smoothing is applied.

    Returns
    -------
    nshift : int
    """
    if smooth_kwargs is None:
        return 0
    else:
        return ceil(2 * smooth_kwargs["sigma"])


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


def get_halolims(pos, ncells, nshift=None):
    """
    Get the lower and upper limit of a halo's positions or cell numbers.

    Parameters
    ----------
    pos : 2-dimensional array
        Halo particle array. Columns must be `x`, `y`, `z`.
    ncells : int
        Number of grid cells of the box along a single dimension.
    nshift : int, optional
        Lower and upper shift of the clump limits.

    Returns
    -------
    mins, maxs : 1-dimensional arrays of shape `(3, )`
        Minimum and maximum along each axis.
    """
    # Check that in case of `nshift` we have integer positions.
    dtype = pos.dtype
    if nshift is not None and dtype.char not in numpy.typecodes["AllInteger"]:
        raise TypeError("`nshift` supported only positions are cells.")
    nshift = 0 if nshift is None else nshift  # To simplify code below

    mins = numpy.full(3, numpy.nan, dtype=dtype)
    maxs = numpy.full(3, numpy.nan, dtype=dtype)
    for i in range(3):
        mins[i] = max(numpy.min(pos[:, i]) - nshift, 0)
        maxs[i] = min(numpy.max(pos[:, i]) + nshift, ncells - 1)

    return mins, maxs


@jit(nopython=True)
def calculate_overlap(delta1, delta2, cellmins, delta_bckg):
    r"""
    Overlap between two halos whose density fields are evaluated on the
    same grid. This is a JIT implementation, hence it is outside of the main
    class.

    Parameters
    ----------
    delta1: 3-dimensional array
        Density field of the first halo.
    delta2 : 3-dimensional array
        Density field of the second halo.
    cellmins : len-3 tuple
        Tuple of left-most cell ID in the full box.
    delta_bckg : 3-dimensional array
        Summed background density field of the reference and cross simulations
        calculated with particles assigned to halos at the final snapshot.
        Assumed to only be sampled in cells :math:`[512, 1536)^3`.

    Returns
    -------
    overlap : float
    """
    totmass = 0.0  # Total mass of clump 1 and clump 2
    intersect = 0.0  # Weighted intersecting mass
    i0, j0, k0 = cellmins  # Unpack things
    bckg_size = 2 * BCKG_HALFSIZE
    bckg_offset = BOX_SIZE // 2 - BCKG_HALFSIZE
    imax, jmax, kmax = delta1.shape

    for i in range(imax):
        ii = i0 + i - bckg_offset
        ishighres = 0 <= ii < bckg_size
        for j in range(jmax):
            jj = j0 + j - bckg_offset
            ishighres &= 0 <= jj < bckg_size
            for k in range(kmax):
                kk = k0 + k - bckg_offset
                ishighres &= 0 <= kk < bckg_size

                m1, m2 = delta1[i, j, k], delta2[i, j, k]
                totmass += m1 + m2
                prod = 2 * m1 * m2
                if prod > 0:  # If both cells are non-zero
                    bcgk = delta_bckg[ii, jj, kk] if ishighres else m1 + m2
                    intersect += prod / bcgk if bcgk > 0 else prod / (m1 + m2)

    return intersect / (totmass - intersect)


@jit(nopython=True)
def calculate_overlap_indxs(delta1, delta2, cellmins, delta_bckg, nonzero,
                            mass1, mass2):
    r"""
    Overlap between two clumps whose density fields are evaluated on the
    same grid and `nonzero1` enumerates the non-zero cells of `delta1.  This is
    a JIT implementation, hence it is outside of the main class.

    Parameters
    ----------
    delta1: 3-dimensional array
        Density field of the first halo.
    delta2 : 3-dimensional array
        Density field of the second halo.
    cellmins : len-3 tuple
        Tuple of left-most cell ID in the full box.
    delta_bckg : 3-dimensional array
        Summed background density field of the reference and cross simulations
        calculated with particles assigned to halos at the final snapshot.
        Assumed to only be sampled in cells :math:`[512, 1536)^3`.
    nonzero : 2-dimensional array of shape `(n_cells, 3)`
        Indices of cells that are non-zero of the lower mass clump. Expected to
        be precomputed from `fill_delta_indxs`.
    mass1, mass2 : floats, optional
        Total masses of the two clumps, respectively. Optional. If not provided
        calculcated directly from the density field.

    Returns
    -------
    overlap : float
    """
    intersect = 0.0  # Weighted intersecting mass
    i0, j0, k0 = cellmins  # Unpack cell minimas
    bckg_size = 2 * BCKG_HALFSIZE
    bckg_offset = BOX_SIZE // 2 - BCKG_HALFSIZE

    for n in range(nonzero.shape[0]):
        i, j, k = nonzero[n, :]
        m1, m2 = delta1[i, j, k], delta2[i, j, k]
        prod = 2 * m1 * m2

        if prod > 0:
            ii = i0 + i - bckg_offset  # Indices of this cell in the
            jj = j0 + j - bckg_offset  # background density field.
            kk = k0 + k - bckg_offset

            ishighres = 0 <= ii < bckg_size  # Is this cell is in the high
            ishighres &= 0 <= jj < bckg_size  # resolution region for which the
            ishighres &= 0 <= kk < bckg_size  # background field is calculated.

            bckg = delta_bckg[ii, jj, kk] if ishighres else m1 + m2
            intersect += prod / bckg if bckg > 0 else prod / (m1 + m2)

    return intersect / (mass1 + mass2 - intersect)


def load_processed_halo(hid, particles, clump_map, clid2map, clumps_cat,
                        ncells, nshift):
    """
    Load a processed halo from the `.h5` file. This is to be wrapped by a
    cacher.

    Parameters
    ----------
    hid : int
        Halo ID.
    particles : 2-dimensional array
        Array of particles in box units. The columns must be `x`, `y`, `z`
        and `M`.
    clump_map : 2-dimensional array
        Array containing start and end indices in the particle array
        corresponding to each clump.
    clid2map : dict
        Dictionary mapping clump IDs to `clump_map` array positions.
    clumps_cat : :py:class:`csiborgtools.read.ClumpsCatalogue`
        Clumps catalogue.
    ncells : int
        Number of cells in the original density field. Typically 2048.
    nshift : int
        Number of cells to pad the density field.

    Returns
    -------
    pos : 2-dimensional array
        Array of cell particle positions.
    mass : 1-dimensional array
        Array of particle masses.
    totmass : float
        Total mass of the halo.
    mins : len-3 tuple
        Minimum cell indices of the halo.
    maxs : len-3 tuple
        Maximum cell indices of the halo.
    """
    pos = load_parent_particles(hid, particles, clump_map, clid2map,
                                clumps_cat)
    pos, mass = pos[:, :3], pos[:, 3]
    pos = pos2cell(pos, ncells)
    totmass = numpy.sum(mass)
    mins, maxs = get_halolims(pos, ncells=ncells, nshift=nshift)
    return pos, mass, totmass, mins, maxs


def radius_neighbours(knn, X, radiusX, radiusKNN, nmult=1.0,
                      enforce_int32=False, verbose=True):
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
    enforce_int32 : bool, optional
        Whether to enforce 32-bit integer precision of the indices. By default
        `False`.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    indxs : 1-dimensional array `(n_samples,)` of arrays
        Matches to `X` from `knn`.
    """
    assert X.ndim == 2 and X.shape[1] == 3  # shape of X ok?
    assert X.shape[0] == radiusX.size  # patchX matches X?
    assert radiusKNN.size == knn.n_samples_fit_  # patchknn matches the knn?

    nsamples = X.shape[0]
    indxs = [None] * nsamples
    patchknn_max = numpy.max(radiusKNN)  # Maximum for completeness

    for i in trange(nsamples) if verbose else range(nsamples):
        dist, indx = knn.radius_neighbors(
            X[i, :].reshape(-1, 3), radiusX[i] + patchknn_max,
            sort_results=True)
        # Note that `dist` and `indx` are wrapped in 1-element arrays
        # so we take the first item where appropriate
        mask = (dist[0] / (radiusX[i] + radiusKNN[indx[0]])) < nmult
        indxs[i] = indx[0][mask]
        if enforce_int32:
            indxs[i] = indxs[i].astype(numpy.int32)

    return numpy.asarray(indxs, dtype=object)
