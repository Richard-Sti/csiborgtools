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

import numpy
from numba import jit
from scipy.ndimage import gaussian_filter
from tqdm import tqdm, trange

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

    def cross(self, cat0, catx, halos0_archive, halosx_archive, delta_bckg,
              verbose=True):
        r"""
        Find all neighbours whose CM separation is less than `nmult` times the
        sum of their initial Lagrangian patch sizes and calculate their
        overlap. Enforces that the neighbours' are similar in mass up to
        `dlogmass` dex.

        Parameters
        ----------
        cat0 : :py:class:`csiborgtools.read.HaloCatalogue`
            Halo catalogue of the reference simulation.
        catx : :py:class:`csiborgtools.read.HaloCatalogue`
            Halo catalogue of the cross simulation.
        halos0_archive : `NpzFile` object
            Archive of halos' particles of the reference simulation, keys must
            include `x`, `y`, `z` and `M`. The positions must already be
            converted to cell numbers.
        halosx_archive : `NpzFile` object
            Archive of halos' particles of the cross simulation, keys must
            include `x`, `y`, `z` and `M`. The positions must already be
            converted to cell numbers.
        delta_bckg : 3-dimensional array
            Summed background density field of the reference and cross
            simulations calculated with particles assigned to halos at the
            final snapshot. Assumed to only be sampled in cells
            :math:`[512, 1536)^3`.
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
            now = datetime.now()
            print(f"{now}: querying the KNN.", flush=True)
        match_indxs = radius_neighbours(
            catx.knn(select_initial=True), cat0.positions(in_initial=True),
            radiusX=cat0["lagpatch"], radiusKNN=catx["lagpatch"],
            nmult=self.nmult, enforce_int32=True, verbose=verbose)
        # We next remove neighbours whose mass is too large/small.
        if self.dlogmass is not None:
            for i, indx in enumerate(match_indxs):
                # |log(M1 / M2)|
                p = self.mass_kind
                aratio = numpy.abs(numpy.log10(catx[p][indx] / cat0[p][i]))
                match_indxs[i] = match_indxs[i][aratio < self.dlogmass]

        # We will make a dictionary to keep in memory the halos' particles from
        # the cross simulations so that they are not loaded in several times
        # and we only convert their positions to cells once. Possibly make an
        # option to not do this to lower memory requirements?
        cross_halos = {}
        cross_lims = {}
        cross = [numpy.asanyarray([], dtype=numpy.float32)] * match_indxs.size
        indxs = cat0["index"]
        for i, k0 in enumerate(tqdm(indxs) if verbose else indxs):
            # If we have no matches continue to the next halo.
            matches = match_indxs[i]
            if matches.size == 0:
                continue
            # Next, we find this halo's particles, total mass, minimum and
            # maximum cells and convert positions to cells.
            halo0 = halos0_archive[str(k0)]
            mass0 = numpy.sum(halo0["M"])
            mins0, maxs0 = get_halolims(halo0,
                                        ncells=self.overlapper.inv_clength,
                                        nshift=self.overlapper.nshift)
            for p in ("x", "y", "z"):
                halo0[p] = self.overlapper.pos2cell(halo0[p])
            # We now loop over matches of this halo and calculate their
            # overlap, storing them in `_cross`.
            _cross = numpy.full(matches.size, numpy.nan, dtype=numpy.float32)
            for j, kf in enumerate(catx["index"][matches]):
                # Attempt to load this cross halo from memory, if it fails get
                # it from from the halo archive (and similarly for the limits)
                # and convert the particle positions to cells.
                try:
                    halox = cross_halos[kf]
                    minsx, maxsx = cross_lims[kf]
                except KeyError:
                    halox = halosx_archive[str(kf)]
                    minsx, maxsx = get_halolims(
                        halox, ncells=self.overlapper.inv_clength,
                        nshift=self.overlapper.nshift)
                    for p in ("x", "y", "z"):
                        halox[p] = self.overlapper.pos2cell(halox[p])
                    cross_halos[kf] = halox
                    cross_lims[kf] = (minsx, maxsx)
                massx = numpy.sum(halox["M"])
                _cross[j] = self.overlapper(halo0, halox, delta_bckg, mins0,
                                            maxs0, minsx, maxsx, mass1=mass0,
                                            mass2=massx)
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

    def smoothed_cross(self, cat0, catx, halos0_archive, halosx_archive,
                       delta_bckg, match_indxs, smooth_kwargs, verbose=True):
        r"""
        Calculate the smoothed overlaps for pair previously identified via
        `self.cross(...)` to have a non-zero overlap.

        Parameters
        ----------
        cat0 : :py:class:`csiborgtools.read.ClumpsCatalogue`
            Halo catalogue of the reference simulation.
        catx : :py:class:`csiborgtools.read.ClumpsCatalogue`
            Halo catalogue of the cross simulation.
        halos0_archive : `NpzFile` object
            Archive of halos' particles of the reference simulation, keys must
            include `x`, `y`, `z` and `M`. The positions must already be
            converted to cell numbers.
        halosx_archive : `NpzFile` object
            Archive of halos' particles of the cross simulation, keys must
            include `x`, `y`, `z` and `M`. The positions must already be
            converted to cell numbers.
        delta_bckg : 3-dimensional array
            Smoothed summed background density field of the reference and cross
            simulations calculated with particles assigned to halos at the
            final snapshot. Assumed to only be sampled in cells
            :math:`[512, 1536)^3`.
        ref_indxs : 1-dimensional array
            Halo IDs in the reference catalogue.
        cross_indxs : 1-dimensional array
            Halo IDs in the cross catalogue.
        match_indxs : 1-dimensional array of arrays
            Indices of halo counterparts in the cross catalogue.
        smooth_kwargs : kwargs
            Kwargs to be passed to :py:func:`scipy.ndimage.gaussian_filter`.
        verbose : bool, optional
            Iterator verbosity flag. By default `True`.

        Returns
        -------
        overlaps : 1-dimensional array of arrays
        """

        cross_halos = {}
        cross_lims = {}
        cross = [numpy.asanyarray([], dtype=numpy.float32)] * match_indxs.size

        indxs = cat0["index"]
        for i, k0 in enumerate(tqdm(indxs) if verbose else indxs):
            halo0 = halos0_archive[str(k0)]
            mins0, maxs0 = get_halolims(halo0,
                                        ncells=self.overlapper.inv_clength,
                                        nshift=self.overlapper.nshift)

            # Now loop over the matches and calculate the smoothed overlap.
            _cross = numpy.full(match_indxs[i].size, numpy.nan, numpy.float32)
            for j, kf in enumerate(catx["index"][match_indxs[i]]):
                # Attempt to load this cross halo from memory, if it fails get
                # it from from the halo archive (and similarly for the limits).
                try:
                    halox = cross_halos[kf]
                    minsx, maxsx = cross_lims[kf]
                except KeyError:
                    halox = halosx_archive[str(kf)]
                    minsx, maxsx = get_halolims(
                        halox, ncells=self.overlapper.inv_clength,
                        nshift=self.overlapper.nshift)
                    cross_halos[kf] = halox
                    cross_lims[kf] = (minsx, maxsx)

                _cross[j] = self.overlapper(halo0, halox, delta_bckg, mins0,
                                            maxs0, minsx, maxsx,
                                            smooth_kwargs=smooth_kwargs)
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

    def __init__(self):
        # Inverse cell length in box units. By default :math:`2^11`, which
        # matches the initial RAMSES grid resolution.
        self.inv_clength = 2**11
        self.nshift = 5  # Hardcode this too to force consistency
        self._clength = 1 / self.inv_clength

    def pos2cell(self, pos):
        """
        Convert position to cell number. If `pos` is in
        `numpy.typecodes["AllInteger"]` assumes it to already be the cell
        number.

        Parameters
        ----------
        pos : 1-dimensional array
            Array of positions along an axis in the box.

        Returns
        -------
        cells : 1-dimensional array
        """
        # Check whether this is already the cell
        if pos.dtype.char in numpy.typecodes["AllInteger"]:
            return pos
        return numpy.floor(pos * self.inv_clength).astype(numpy.int32)

    def make_bckg_delta(self, halo_archive, delta=None, verbose=False):
        """
        Calculate a NGP density field of particles belonging to halos within
        the central :math:`1/2^3` high-resolution region of the simulation.
        Smoothing must be applied separately.

        Parameters
        ----------
        halo_archive : `NpzFile` object
            Archive of halos' particles of the reference simulation, keys must
            include `x`, `y`, `z` and `M`.
        delta : 3-dimensional array, optional
            Array to store the density field in. If `None` a new array is
            created.
        verbose : bool, optional
            Verbosity flag for loading the files.

        Returns
        -------
        delta : 3-dimensional array
        """
        # We obtain the minimum/maximum cell IDs and number of cells
        cellmin = self.inv_clength // 4  # The minimum cell ID
        cellmax = 3 * self.inv_clength // 4  # The maximum cell ID
        ncells = cellmax - cellmin
        # We then pre-allocate the density field/check it is of the right shape
        if delta is None:
            delta = numpy.zeros((ncells,) * 3, dtype=numpy.float32)
        else:
            assert ((delta.shape == (ncells,) * 3)
                    & (delta.dtype == numpy.float32))

        # We now loop one-by-one over the halos fill the density field.
        files = halo_archive.files
        for file in tqdm(files) if verbose else files:
            parts = halo_archive[file]
            cells = [self.pos2cell(parts[p]) for p in ("x", "y", "z")]
            mass = parts["M"]

            # We mask out particles outside the cubical high-resolution region
            mask = ((cellmin <= cells[0])
                    & (cells[0] < cellmax)
                    & (cellmin <= cells[1])
                    & (cells[1] < cellmax)
                    & (cellmin <= cells[2])
                    & (cells[2] < cellmax))
            cells = [c[mask] for c in cells]
            mass = mass[mask]
            fill_delta(delta, *cells, *(cellmin,) * 3, mass)

        return delta

    def make_delta(self, clump, mins=None, maxs=None, subbox=False,
                   smooth_kwargs=None):
        """
        Calculate a NGP density field of a halo on a cubic grid. Optionally can
        be smoothed with a Gaussian kernel.

        Parameters
        ----------
        clump : structurered arrays
            Clump structured array, keys must include `x`, `y`, `z` and `M`.
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
        cells = [self.pos2cell(clump[p]) for p in ("x", "y", "z")]

        # Check that minima and maxima are integers
        if not (mins is None and maxs is None):
            assert mins.dtype.char in numpy.typecodes["AllInteger"]
            assert maxs.dtype.char in numpy.typecodes["AllInteger"]

        if subbox:
            # Minimum xcell, ycell and zcell of this clump
            if mins is None or maxs is None:
                mins = numpy.asanyarray(
                    [max(numpy.min(cell) - self.nshift, 0) for cell in cells]
                )
                maxs = numpy.asanyarray(
                    [
                        min(numpy.max(cell) + self.nshift, self.inv_clength)
                        for cell in cells
                    ]
                )

            ncells = numpy.max(maxs - mins) + 1  # To get the number of cells
        else:
            mins = [0, 0, 0]
            ncells = self.inv_clength

        # Preallocate and fill the array
        delta = numpy.zeros((ncells,) * 3, dtype=numpy.float32)
        fill_delta(delta, *cells, *mins, clump["M"])

        if smooth_kwargs is not None:
            gaussian_filter(delta, output=delta, **smooth_kwargs)
        return delta

    def make_deltas(self, clump1, clump2, mins1=None, maxs1=None, mins2=None,
                    maxs2=None, smooth_kwargs=None):
        """
        Calculate a NGP density fields of two halos on a grid that encloses
        them both. Optionally can be smoothed with a Gaussian kernel.

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
        xc1, yc1, zc1 = (self.pos2cell(clump1[p]) for p in ("x", "y", "z"))
        xc2, yc2, zc2 = (self.pos2cell(clump2[p]) for p in ("x", "y", "z"))

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
            xmax, ymax, zmax = [
                min(px, self.inv_clength - 1) for px in (xmax, ymax, zmax)
            ]
        else:
            xmin, ymin, zmin = [min(mins1[i], mins2[i]) for i in range(3)]
            xmax, ymax, zmax = [max(maxs1[i], maxs2[i]) for i in range(3)]

        cellmins = (xmin, ymin, zmin)  # Cell minima
        ncells = max(xmax - xmin, ymax - ymin, zmax - zmin) + 1  # Num cells

        # Preallocate and fill the arrays
        delta1 = numpy.zeros((ncells,) * 3, dtype=numpy.float32)
        delta2 = numpy.zeros((ncells,) * 3, dtype=numpy.float32)

        # If no smoothing figure out the nonzero indices of the smaller clump
        if smooth_kwargs is None:
            if clump1.size > clump2.size:
                fill_delta(delta1, xc1, yc1, zc1, *cellmins, clump1["M"])
                nonzero = fill_delta_indxs(
                    delta2, xc2, yc2, zc2, *cellmins, clump2["M"]
                )
            else:
                nonzero = fill_delta_indxs(
                    delta1, xc1, yc1, zc1, *cellmins, clump1["M"]
                )
                fill_delta(delta2, xc2, yc2, zc2, *cellmins, clump2["M"])
        else:
            fill_delta(delta1, xc1, yc1, zc1, *cellmins, clump1["M"])
            fill_delta(delta2, xc2, yc2, zc2, *cellmins, clump2["M"])
            nonzero = None

        if smooth_kwargs is not None:
            gaussian_filter(delta1, output=delta1, **smooth_kwargs)
            gaussian_filter(delta2, output=delta2, **smooth_kwargs)
        return delta1, delta2, cellmins, nonzero

    def __call__(self, clump1, clump2, delta_bckg, mins1=None, maxs1=None,
                 mins2=None, maxs2=None, mass1=None, mass2=None,
                 smooth_kwargs=None):
        """
        Calculate overlap between `clump1` and `clump2`. See
        `calculate_overlap(...)` for further information. Be careful so that
        the background density field is calculated with the same
        `smooth_kwargs`. If any smoothing is applied then loops over the full
        density fields, otherwise only over the non-zero cells of the lower
        mass clump.

        Parameters
        ----------
        clump1, clump2 : structurered arrays
            Structured arrays containing the particles of a given clump. Keys
            must include `x`, `y`, `z` and `M`.
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
        mass1, mass2 : floats, optional
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
            clump1, clump2, mins1, maxs1, mins2, maxs2,
            smooth_kwargs=smooth_kwargs)

        if smooth_kwargs is not None:
            return calculate_overlap(delta1, delta2, cellmins, delta_bckg)
        # Calculate masses not given
        mass1 = numpy.sum(clump1["M"]) if mass1 is None else mass1
        mass2 = numpy.sum(clump2["M"]) if mass2 is None else mass2
        return calculate_overlap_indxs(
            delta1, delta2, cellmins, delta_bckg, nonzero, mass1, mass2)


###############################################################################
#                     Halo matching supplementary functions                   #
###############################################################################


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


def get_halolims(halo, ncells, nshift=None):
    """
    Get the lower and upper limit of a halo's positions or cell numbers.

    Parameters
    ----------
    halo : structured array
        Structured array containing the particles of a given halo. Keys must
        `x`, `y`, `z`.
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
    dtype = halo["x"].dtype
    if nshift is not None and dtype.char not in numpy.typecodes["AllInteger"]:
        raise TypeError("`nshift` supported only positions are cells.")
    nshift = 0 if nshift is None else nshift  # To simplify code below

    mins = numpy.full(3, numpy.nan, dtype=dtype)
    maxs = numpy.full(3, numpy.nan, dtype=dtype)

    for i, p in enumerate(["x", "y", "z"]):
        mins[i] = max(numpy.min(halo[p]) - nshift, 0)
        maxs[i] = min(numpy.max(halo[p]) + nshift, ncells - 1)

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
    bckg_offset = 512  # Offset of the background density field
    bckg_size = 1024
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
    bckg_offset = 512  # Offset of the background density field
    bckg_size = 1024  # Size of the background density field array

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


def dist_centmass(clump):
    """
    Calculate the clump (or halo) particles' distance from the centre of mass.

    Parameters
    ----------
    clump : 2-dimensional array of shape (n_particles, 7)
        Particle array. The first four columns must be `x`, `y`, `z` and `M`.

    Returns
    -------
    dist : 1-dimensional array of shape `(n_particles, )`
        Particle distance from the centre of mass.
    cm : 1-dimensional array of shape `(3,)`
        Center of mass coordinates.
    """
    # CM along each dimension
    cm = numpy.average(clump[:, :3], weights=clump[:, 3], axis=0)
    return numpy.linalg.norm(clump[:, :3] - cm, axis=1)


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
