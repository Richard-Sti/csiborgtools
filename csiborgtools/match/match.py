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
Support for matching halos between CSiBORG IC realisations based on their
Lagrangian patch overlap.
"""
from abc import ABC
from datetime import datetime
from functools import lru_cache
from math import ceil

import numpy
from numba import jit
from scipy.ndimage import gaussian_filter
from tqdm import tqdm, trange


class BaseMatcher(ABC):
    """Base class for `RealisationsMatcher` and `ParticleOverlap`."""
    _box_size = None
    _bckg_halfsize = None

    @property
    def box_size(self):
        """Number of cells in the box."""
        if self._box_size is None:
            raise RuntimeError("`box_size` has not been set.")
        return self._box_size

    @box_size.setter
    def box_size(self, value):
        if not (isinstance(value, int) and value > 0):
            raise ValueError("`box_size` must be a positive integer.")
        if not value != 0 and (value & (value - 1) == 0):
            raise ValueError("`box_size` must be a power of 2.")
        self._box_size = value

    @property
    def bckg_halfsize(self):
        """
        Background half-size for density field calculation. This is the
        grid distance from the center of the box to each side over which to
        evaluate the background density field. Must be less than or equal to
        half the box size.
        """
        if self._bckg_halfsize is None:
            raise RuntimeError("`bckg_halfsize` has not been set.")
        return self._bckg_halfsize

    @bckg_halfsize.setter
    def bckg_halfsize(self, value):
        if not (isinstance(value, int) and value > 0):
            raise ValueError("`bckg_halfsize` must be a positive integer.")
        if value > self.box_size // 2:
            raise ValueError("`bckg_halfsize` must be <= half the box size.")
        self._bckg_halfsize = value


###############################################################################
#                  Realisations matcher for calculating overlaps              #
###############################################################################


class RealisationsMatcher(BaseMatcher):
    """
    Matches haloes between IC realisations.

    Parameters
    ----------
    box_size : int
        Number of cells in the box.
    bckg_halfsize : int
        Background half-size for density field calculation. This is the
        grid distance from the center of the box to each side over which to
        evaluate the background density field. Must be less than or equal to
        half the box size.
    nmult : float or int, optional
        Multiplier of the sum of the initial Lagrangian patch sizes of a halo
        pair. Determines the range within which neighbors are returned.
    dlogmass : float, optional
        Tolerance on the absolute logarithmic mass difference of potential
        matches.
    """
    _nmult = None
    _dlogmass = None
    _mass_key = None
    _overlapper = None

    def __init__(self, box_size, bckg_halfsize, nmult=1.0, dlogmass=2.0):
        self.box_size = box_size
        self.bckg_halfsize = bckg_halfsize
        self.nmult = nmult
        self.dlogmass = dlogmass
        self.mass_key = "totmass"

        self._overlapper = ParticleOverlap(box_size, bckg_halfsize)

    @property
    def nmult(self):
        """
        Multiplier of the sum of the initial Lagrangian patch sizes of a halo
        pair. Determines the range within which neighbors are returned.

        Returns
        -------
        float
        """
        return self._nmult

    @nmult.setter
    def nmult(self, value):
        if not (value > 0 and isinstance(value, (int, float))):
            raise ValueError("`nmult` must be a positive integer or float.")
        self._nmult = float(value)

    @property
    def dlogmass(self):
        """
        Tolerance on the absolute logarithmic mass difference of potential
        matches.

        Returns
        -------
        float
        """
        return self._dlogmass

    @dlogmass.setter
    def dlogmass(self, value):
        if not (value > 0 and isinstance(value, (float, int))):
            raise ValueError("`dlogmass` must be a positive float.")
        self._dlogmass = float(value)

    @property
    def mass_key(self):
        """
        Mass key whose similarity is to be checked. Must be a valid key in the
        halo catalogue.

        Returns
        -------
        str
        """
        return self._mass_key

    @mass_key.setter
    def mass_key(self, value):
        if not isinstance(value, str):
            raise ValueError("`mass_key` must be a string.")
        self._mass_key = value

    @property
    def overlapper(self):
        """The overlapper object."""
        return self._overlapper

    def cross(self, cat0, catx, delta_bckg, cache_size=10000, verbose=True):
        r"""
        Find all neighbours whose CM separation is less than `nmult` times the
        sum of their initial Lagrangian patch sizes and calculate their
        overlap. Enforces that the neighbours are similar in mass up to
        `dlogmass` dex.

        Parameters
        ----------
        cat0 : instance of :py:class:`csiborgtools.read.BaseCatalogue`
            Halo catalogue of the reference simulation.
        catx : instance of :py:class:`csiborgtools.read.BaseCatalogue`
            Halo catalogue of the cross simulation.
        delta_bckg : 3-dimensional array
            Summed background density field of the reference and cross
            simulations calculated with particles assigned to halos at the
            final snapshot. Calculated on a grid determined by `bckg_halfsize`.
        cache_size : int, optional
            Caching size for loading the cross simulation halos.
        verbose : bool, optional
            Iterator verbosity flag. By default `true`.

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
        match_indxs = radius_neighbours(
            catx.knn(in_initial=True, subtract_observer=False, periodic=True),
            cat0["lagpatch_coordinates"], radiusX=cat0["lagpatch_radius"],
            radiusKNN=catx["lagpatch_radius"], nmult=self.nmult,
            enforce_int32=True, verbose=verbose)

        # We next remove neighbours whose mass is too large/small.
        if self.dlogmass is not None:
            for i, indx in enumerate(match_indxs):
                # |log(M1 / M2)|
                p = self.mass_key
                aratio = numpy.abs(numpy.log10(catx[p][indx] / cat0[p][i]))
                match_indxs[i] = match_indxs[i][aratio < self.dlogmass]

        # We will cache the halos from the cross simulation to speed up the I/O
        @lru_cache(maxsize=cache_size)
        def load_cached_halox(hid):
            return load_processed_halo(hid, catx, nshift=0,
                                       ncells=self.box_size)

        iterator = tqdm(
            cat0["index"],
            desc=f"{datetime.now()}: calculating NGP overlaps",
            disable=not verbose
            )
        cross = [numpy.asanyarray([], dtype=numpy.float32)] * match_indxs.size
        for i, k0 in enumerate(iterator):
            # If we have no matches continue to the next halo.
            matches = match_indxs[i]
            if matches.size == 0:
                continue
            # Next, we find this halo's particles, total mass, minimum and
            # maximum cells and convert positions to cells.
            pos0, mass0, totmass0, mins0, maxs0 = load_processed_halo(
                k0, cat0, nshift=0, ncells=self.box_size)

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

    def smoothed_cross(self, cat0, catx, delta_bckg, match_indxs,
                       smooth_kwargs, cache_size=10000, verbose=True):
        r"""
        Calculate the smoothed overlaps for pairs previously identified via
        `self.cross(...)` to have a non-zero NGP overlap.

        Parameters
        ----------
        cat0 : instance of :py:class:`csiborgtools.read.BaseCatalogue`
            Halo catalogue of the reference simulation.
        catx : instance of :py:class:`csiborgtools.read.BaseCatalogue`
            Halo catalogue of the cross simulation.
        delta_bckg : 3-dimensional array
            Smoothed summed background density field of the reference and cross
            simulations calculated with particles assigned to halos at the
            final snapshot. Calculated on a grid determined by `bckg_halfsize`.
        match_indxs : 1-dimensional array of arrays
            Indices of halo counterparts in the cross catalogue.
        smooth_kwargs : kwargs
            Kwargs to be passed to :py:func:`scipy.ndimage.gaussian_filter`.
        cache_size : int, optional
            Caching size for loading the cross simulation halos.
        verbose : bool, optional
            Iterator verbosity flag. By default `true`.

        Returns
        -------
        overlaps : 1-dimensional array of arrays
        """
        nshift = read_nshift(smooth_kwargs)

        @lru_cache(maxsize=cache_size)
        def load_cached_halox(hid):
            return load_processed_halo(hid, catx, nshift=nshift,
                                       ncells=self.box_size)

        iterator = tqdm(
            cat0["index"],
            desc=f"{datetime.now()}: calculating smoothed overlaps",
            disable=not verbose
            )
        cross = [numpy.asanyarray([], dtype=numpy.float32)] * match_indxs.size
        for i, k0 in enumerate(iterator):
            pos0, mass0, __, mins0, maxs0 = load_processed_halo(
                k0, cat0, nshift=nshift, ncells=self.box_size)

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
#                       Overlap calculator                                    #
###############################################################################


class ParticleOverlap(BaseMatcher):
    r"""
    Halo overlaps calculator. The density field calculation is based on the
    nearest grid position particle assignment scheme, with optional Gaussian
    smoothing.

    Parameters
    ----------
    box_size : int
        Number of cells in the box.
    bckg_halfsize : int
        Background half-size for density field calculation. This is the
        grid distance from the center of the box to each side over which to
        evaluate the background density field. Must be less than or equal to
        half the box size.
    """

    def __init__(self, box_size, bckg_halfsize):
        self.box_size = box_size
        self.bckg_halfsize = bckg_halfsize

    def make_bckg_delta(self, cat, delta=None, verbose=False):
        """
        Calculate a NGP density field of particles belonging to halos of a
        halo catalogue `halo_cat`. Particles are only counted within the
        high-resolution region of the simulation. Smoothing must be applied
        separately.

        Parameters
        ----------
        cat : instance of :py:class:`csiborgtools.read.BaseCatalogue`
            Halo catalogue of the reference simulation.
        delta : 3-dimensional array, optional
            Array to store the density field. If `None` a new array is
            created.
        verbose : bool, optional
            Verbosity flag for loading the halos' particles.

        Returns
        -------
        delta : 3-dimensional array
        """
        cellmin = self.box_size // 2 - self.bckg_halfsize
        cellmax = self.box_size // 2 + self.bckg_halfsize
        ncells = cellmax - cellmin
        # We then pre-allocate the density field/check it is of the right shape
        if delta is None:
            delta = numpy.zeros((ncells,) * 3, dtype=numpy.float32)
        else:
            assert ((delta.shape == (ncells,) * 3)
                    & (delta.dtype == numpy.float32))

        iterator = tqdm(
            cat["index"],
            desc=f"{datetime.now()} Calculating the background field",
            disable=not verbose
            )
        for hid in iterator:
            pos = cat.snapshot.halo_coordinates(hid, is_group=True)
            if pos is None:
                continue

            mass = cat.snapshot.halo_masses(hid, is_group=True)

            pos = pos2cell(pos, self.box_size)

            # We mask out particles outside the cubical high-resolution region
            mask = numpy.all((cellmin <= pos) & (pos < cellmax), axis=1)
            pos = pos[mask]
            fill_delta(delta, pos[:, 0], pos[:, 1], pos[:, 2],
                       *(cellmin,) * 3, mass[mask])
        return delta

    def make_delta(self, pos, mass, mins=None, maxs=None, subbox=False,
                   smooth_kwargs=None):
        """
        Calculate a NGP density field of a halo on a regular grid. Optionally
        can be smoothed with a Gaussian kernel.

        Parameters
        ----------
        pos : 2-dimensional array
            Halo's particles position array.
        mass : 1-dimensional array
            Halo's particles mass array.
        mins, maxs : 1-dimensional arrays of shape `(3,)`
            Minimun and maximum cell numbers along each dimension.
        subbox : bool, optional
            Whether to calculate the field on a grid enclosing the halo.
        smooth_kwargs : kwargs, optional
            Kwargs to be passed to :py:func:`scipy.ndimage.gaussian_filter`.
            If `None` no smoothing is applied.

        Returns
        -------
        delta : 3-dimensional array
        """
        nshift = read_nshift(smooth_kwargs)
        cells = pos2cell(pos, self.box_size)

        if not (mins is None and maxs is None):
            assert mins.dtype.char in numpy.typecodes["AllInteger"]
            assert maxs.dtype.char in numpy.typecodes["AllInteger"]

        if subbox:
            if mins is None or maxs is None:
                mins, maxs = get_halo_cell_limits(cells, self.box_size, nshift)
            ncells = maxs - mins + 1
        else:
            mins = [0, 0, 0]
            ncells = (self.box_size, ) * 3

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
            Minimun and maximum cell numbers along each dimension of `halo1`.
            Optional.
        mins2, maxs2 : 1-dimensional arrays of shape `(3,)`
            Minimum and maximum cell numbers along each dimension of `halo2`.
            Optional.
        smooth_kwargs : kwargs, optional
            Kwargs to be passed to :py:func:`scipy.ndimage.gaussian_filter`.
            If `None` no smoothing is applied.

        Returns
        -------
        delta1, delta2 : 3-dimensional arrays
            Density arrays of `halo1` and `halo2`, respectively.
        cellmins : len-3 tuple
            Tuple of left-most cell ID in the full box.
        nonzero : 2-dimensional array
            Indices where the lower mass halo has a non-zero density.
            Calculated only if no smoothing is applied, otherwise `None`.
        """
        nshift = read_nshift(smooth_kwargs)
        pos1 = pos2cell(pos1, self.box_size)
        pos2 = pos2cell(pos2, self.box_size)

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
            xmax, ymax, zmax = [min(px, self.box_size - 1)
                                for px in (xmax, ymax, zmax)]
        else:
            xmin, ymin, zmin = [min(mins1[i], mins2[i]) for i in range(3)]
            xmax, ymax, zmax = [max(maxs1[i], maxs2[i]) for i in range(3)]

        cellmins = (xmin, ymin, zmin)
        ncells = (xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1,)

        delta1 = numpy.zeros(ncells, dtype=numpy.float32)
        delta2 = numpy.zeros(ncells, dtype=numpy.float32)

        # If no smoothing figure out the nonzero indices of the smaller halo
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
        Calculate overlap between `halo1` and `halo2`. See
        `calculate_overlap(...)` for further information. Be careful so that
        the background density field is calculated with the same
        `smooth_kwargs`. If any smoothing is applied then loops over the full
        density fields, otherwise only over the non-zero cells of the lower
        mass halo.

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
            Minimun and maximum cell numbers along each dimension of `halo1`.
            Optional.
        mins2, maxs2 : 1-dimensional arrays of shape `(3,)`
            Minimum and maximum cell numbers along each dimension of `halo2`,
            optional.
        totmass1, totmass2 : floats, optional
            Total mass of `halo1` and `halo2`, respectively. Must be provided
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
            return calculate_overlap(delta1, delta2, cellmins, delta_bckg,
                                     self.box_size, self.bckg_halfsize)

        totmass1 = numpy.sum(mass1) if totmass1 is None else totmass1
        totmass2 = numpy.sum(mass2) if totmass2 is None else totmass2

        return calculate_overlap_indxs(delta1, delta2, cellmins, delta_bckg,
                                       nonzero, totmass1, totmass2,
                                       self.box_size, self.bckg_halfsize)


###############################################################################
#                     Halo matching supplementary functions                   #
###############################################################################


def pos2cell(pos, ncells):
    if pos.dtype.char in numpy.typecodes["AllInteger"]:
        return pos
    return numpy.floor(pos * ncells).astype(numpy.int32)


def read_nshift(smooth_kwargs):
    return 0 if smooth_kwargs is None else ceil(3 * smooth_kwargs["sigma"])


@jit(nopython=True)
def fill_delta(delta, xcell, ycell, zcell, xmin, ymin, zmin, weights):
    """
    Fill array `delta` by adding `weights` to the specified cells.

    Parameters
    ----------
    delta : 3-dimensional array
        Grid to be filled with weights.
    xcell, ycell, zcell : 1-dimensional arrays
        Indices where to assign `weights`.
    xmin, ymin, zmin : ints
        Minimum cell IDs of particles.
    weights : 1-dimensional arrays
        Weights

    Returns
    -------
    None
    """
    n_particles = xcell.size

    for n in range(n_particles):
        i, j, k = xcell[n] - xmin, ycell[n] - ymin, zcell[n] - zmin
        delta[i, j, k] += weights[n]


@jit(nopython=True)
def fill_delta_indxs(delta, xcell, ycell, zcell, xmin, ymin, zmin, weights):
    """
    Fill array `delta` by adding `weights` to the specified cells. Returns
    the indices where `delta` was assigned a value.

    Parameters
    ----------
    delta : 3-dimensional array
        Grid to be filled with weights.
    xcell, ycell, zcell : 1-dimensional arrays
        Indices where to assign `weights`.
    xmin, ymin, zmin : ints
        Minimum cell IDs of particles.
    weights : 1-dimensional arrays
        Weights.

    Returns
    -------
    cells : 1-dimensional array
        Indices where `delta` was assigned a value.
    """
    n_particles = xcell.size
    cells = numpy.full((n_particles, 3), numpy.nan, numpy.int32)
    count_nonzero = 0

    for n in range(n_particles):
        i, j, k = xcell[n] - xmin, ycell[n] - ymin, zcell[n] - zmin

        if delta[i, j, k] == 0:
            cells[count_nonzero] = i, j, k
            count_nonzero += 1

        delta[i, j, k] += weights[n]

    return cells[:count_nonzero]


@jit(nopython=True)
def get_halo_cell_limits(pos, ncells, nshift=0):
    """
    Get the lower and upper limit of a halo's cell numbers. Optionally,
    floating point positions are also supported. However, in this case `nshift`
    must be 0. Be careful, no error will be raised.

    Parameters
    ----------
    pos : 2-dimensional array
        Halo particle array. The first three columns must be the cell numbers
        corresponding to `x`, `y`, `z`.
    ncells : int
        Number of grid cells of the box along a single dimension.
    nshift : int, optional
        Lower and upper shift of the halo limits.

    Returns
    -------
    mins, maxs : 1-dimensional arrays of shape `(3, )`
    """
    dtype = pos.dtype

    mins = numpy.full(3, numpy.nan, dtype=dtype)
    maxs = numpy.full(3, numpy.nan, dtype=dtype)

    for i in range(3):
        mins[i] = max(numpy.min(pos[:, i]) - nshift, 0)
        maxs[i] = min(numpy.max(pos[:, i]) + nshift, ncells - 1)

    return mins, maxs


@jit(nopython=True, boundscheck=False)
def calculate_overlap(delta1, delta2, cellmins, delta_bckg, box_size,
                      bckg_halfsize):
    """
    Calculate overlap between two halos' density fields on the same grid.

    Parameters
    ----------
    delta1, delta2 : 3D array
        Density fields of the first and second halos, respectively.
    cellmins : tuple (len=3)
        Lower cell ID in the full box.
    delta_bckg : 3D array
        Combined background density field of reference and cross simulations
        on `bckg_halfsize` grid.
    box_size : int
        Cell count in the box.
    bckg_halfsize : int
        Grid distance from box center for background density.
        ≤ 0.5 * box_size.

    Returns
    -------
    overlap : float
    """
    totmass = 0.0
    intersect = 0.0
    bckg_size = 2 * bckg_halfsize
    bckg_offset = box_size // 2 - bckg_halfsize

    i0, j0, k0 = cellmins
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


@jit(nopython=True, boundscheck=False)
def calculate_overlap_indxs(delta1, delta2, cellmins, delta_bckg, nonzero,
                            mass1, mass2, box_size, bckg_halfsize):
    """
    Calculate overlap of two halos' density fields on the same grid.

    Parameters
    ----------
    delta1, delta2 : 3D array
        Density fields of the first and second halos, respectively.
    cellmins : tuple (len=3)
        Lower cell ID in the full box.
    delta_bckg : 3D array
        Combined background density from reference and cross simulations
        on `bckg_halfsize` grid.
    nonzero : 2D array (shape: (n_cells, 3))
        Non-zero cells for the lower mass halo (from `fill_delta_indxs`).
    mass1, mass2 : float, optional
        Halos' total masses. Calculated from density if not provided.
    box_size : int
        Cell count in the box.
    bckg_halfsize : int
        Grid distance from box center for background density; ≤ 0.5 * box_size.

    Returns
    -------
    overlap : float
    """
    intersect = 0.0
    bckg_size = 2 * bckg_halfsize
    bckg_offset = box_size // 2 - bckg_halfsize

    i0, j0, k0 = cellmins

    for n in range(nonzero.shape[0]):
        i, j, k = nonzero[n, :]
        m1, m2 = delta1[i, j, k], delta2[i, j, k]
        prod = 2 * m1 * m2

        if prod > 0:
            ii = i0 + i - bckg_offset  # Indices of this cell in the
            jj = j0 + j - bckg_offset  # background density field.
            kk = k0 + k - bckg_offset

            ishighres = 0 <= ii < bckg_size   # Is this cell is in the high
            ishighres &= 0 <= jj < bckg_size  # resolution region for which the
            ishighres &= 0 <= kk < bckg_size  # background field is calculated.

            bckg = delta_bckg[ii, jj, kk] if ishighres else m1 + m2
            intersect += prod / bckg if bckg > 0 else prod / (m1 + m2)

    return intersect / (mass1 + mass2 - intersect)


def load_processed_halo(hid, cat, ncells, nshift):
    """
    Load a processed halo from the `.h5` file. This is to be wrapped by a
    cacher.

    Parameters
    ----------
    hid : int
        Halo ID.
    cat : instance of :py:class:`csiborgtools.read.BaseCatalogue`
        Halo catalogue.
    ncells : int
        Number of cells in the box density field.
    nshift : int
        Cell padding for the density field.

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
    pos = cat.snapshot.halo_coordinates(hid, is_group=True)
    mass = cat.snapshot.halo_masses(hid, is_group=True)

    pos = pos2cell(pos, ncells)
    mins, maxs = get_halo_cell_limits(pos, ncells=ncells, nshift=nshift)
    return pos, mass, numpy.sum(mass), mins, maxs


def radius_neighbours(knn, X, radiusX, radiusKNN, nmult=1.0,
                      enforce_int32=False, verbose=True):
    """
    Find all neigbours of a fitted kNN model whose center of mass separation
    is less than `nmult` times the sum of their respective radii.

    Parameters
    ----------
    knn : :py:class:`sklearn.neighbors.NearestNeighbors`
        Fitted nearest neighbour search.
    X : 2-dimensional array of shape `(n_samples, 3)`
        Array of halo positions from the cross simulation.
    radiusX: 1-dimensional array of shape `(n_samples, )`
        Lagrangian patch radii corresponding to haloes in `X`.
    radiusKNN : 1-dimensional array
        Lagrangian patch radii corresponding to haloes used to train the kNN.
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
    if X.shape != (radiusX.size, 3):
        raise ValueError("Mismatch in shape of `X` or `radiusX`")
    if radiusKNN.size != knn.n_samples_fit_:
        raise ValueError("Mismatch in shape of `radiusKNN` or `knn`")

    patchknn_max = numpy.max(radiusKNN)

    iterator = trange(len(X),
                      desc=f"{datetime.now()}: querying the kNN",
                      disable=not verbose)
    indxs = [None] * len(X)
    for i in iterator:
        dist, indx = knn.radius_neighbors(
            X[i].reshape(1, -1), radiusX[i] + patchknn_max,
            sort_results=True)
        # Note that `dist` and `indx` are wrapped in 1-element arrays
        # so we take the first item where appropriate
        mask = (dist[0] / (radiusX[i] + radiusKNN[indx[0]])) < nmult
        indxs[i] = indx[0][mask]

        if enforce_int32:
            indxs[i] = indxs[i].astype(numpy.int32)

    return numpy.asarray(indxs, dtype=object)


def find_neighbour(nsim0, cats):
    """
    Find the nearest neighbour of halos from a reference catalogue indexed
    `nsim0` in the remaining simulations. Note that this must be the same
    simulation suite.

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
    assert all(isinstance(cat, type(cats[nsim0])) for cat in cats.values())

    cat0 = cats[nsim0]
    X = cat0["lagpatch_coordinates"]

    nhalos = X.shape[0]
    num_cats = len(cats) - 1

    dists = numpy.full((nhalos, num_cats), numpy.nan, dtype=numpy.float32)
    cross_hindxs = numpy.full((nhalos, num_cats), numpy.nan, dtype=numpy.int32)

    # Filter out the reference simulation from the dictionary
    filtered_cats = {k: v for k, v in cats.items() if k != nsim0}

    for i, catx in enumerate(filtered_cats):
        dist, ind = catx.nearest_neighbours(X, radius=1, in_initial=False,
                                            knearest=True)
        dists[:, i] = numpy.ravel(dist)
        cross_hindxs[:, i] = catx["index"][numpy.ravel(ind)]

    return dists, cross_hindxs


###############################################################################
#                     Max's halo matching algorithms                          #
###############################################################################


def matching_max(cat0, catx, mass_key, mult, periodic, overlap=None,
                 match_indxs=None, verbose=True):
    """
    Halo matching algorithm based on [1].

    Parameters
    ----------
    cat0 : instance of :py:class:`csiborgtools.read.BaseCatalogue`
        Halo catalogue of the reference simulation.
    catx : instance of :py:class:`csiborgtools.read.BaseCatalogue`
        Halo catalogue of the cross simulation.
    mass_key : str
        Name of the mass column.
    mult : float
        Multiple of R200c below which to consider a match.
    periodic : bool
        Whether to account for periodic boundary conditions.
    overlap : array of 1-dimensional arrays, optional
        Overlap of halos from `cat0` with halos from `catx`. If `overlap` or
        `match_indxs` is not provided, then the overlap of the identified halos
        is not calculated.
    match_indxs : array of 1-dimensional arrays, optional
        Indicies of halos from `catx` having a non-zero overlap with halos
        from `cat0`.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    out : structured array
        Array of matches. Columns are `hid0`, `hidx`, `dist`, `success`.

    References
    ----------
    [1] Maxwell L Hutt, Harry Desmond, Julien Devriendt, Adrianne Slyz; The
    effect of local Universe constraints on halo abundance and clustering;
    Monthly Notices of the Royal Astronomical Society, Volume 516, Issue 3,
    November 2022, Pages 3592–3601, https://doi.org/10.1093/mnras/stac2407
    """
    pos0 = cat0["cartesian_pos"]
    knnx = catx.knn(in_initial=False, subtract_observer=False,
                    periodic=periodic)
    rad0 = cat0["r200c"]

    mass0 = numpy.log10(cat0[mass_key])
    massx = numpy.log10(catx[mass_key])

    assert numpy.all(numpy.isfinite(mass0)) & numpy.all(numpy.isfinite(massx))

    maskx = numpy.ones(len(catx), dtype=numpy.bool_)

    dtypes = [("hid0", numpy.int32),
              ("hidx", numpy.int32),
              ("mass0", numpy.float32),
              ("massx", numpy.float32),
              ("dist", numpy.float32),
              ("success", numpy.bool_),
              ("match_overlap", numpy.float32),
              ("max_overlap", numpy.float32),
              ]
    out = numpy.full(len(cat0), numpy.nan, dtype=dtypes)
    out["success"] = False

    for i in tqdm(numpy.argsort(mass0)[::-1], desc="Matching haloes",
                  disable=not verbose):
        hid0 = cat0["index"][i]
        out[i]["hid0"] = hid0
        out[i]["mass0"] = 10**mass0[i]

        if not numpy.isfinite(rad0[i]):
            continue

        neigh_dists, neigh_inds = knnx.radius_neighbors(pos0[i].reshape(1, -1),
                                                        mult * rad0[i])
        neigh_dists, neigh_inds = neigh_dists[0], neigh_inds[0]

        if neigh_dists.size == 0:
            continue

        # Sort the neighbours by mass difference
        sort_order = numpy.argsort(numpy.abs(mass0[i] - massx[neigh_inds]))
        neigh_dists = neigh_dists[sort_order]
        neigh_inds = neigh_inds[sort_order]

        for j, neigh_ind in enumerate(neigh_inds):

            if maskx[neigh_ind]:
                out[i]["hidx"] = catx["index"][neigh_ind]
                out[i]["dist"] = neigh_dists[j]
                out[i]["massx"] = 10**massx[neigh_ind]

                out[i]["success"] = True

                maskx[neigh_ind] = False

                if overlap is not None and match_indxs is not None:
                    if neigh_ind in match_indxs[i]:
                        k = numpy.where(neigh_ind == match_indxs[i])[0][0]
                        out[i]["match_overlap"] = overlap[i][k]
                    if len(overlap[i]) > 0:
                        out[i]["max_overlap"] = numpy.max(overlap[i])

                break

    return out
