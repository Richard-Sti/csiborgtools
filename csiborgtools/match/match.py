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
from ..read import concatenate_clumps
from ..utils import now


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
    overlapper_kwargs : dict, optional
        Keyword arguments passed to `ParticleOverlapper`.

    """
    _nmult = None
    _dlogmass = None
    _mass_kind = None
    _overlapper = None

    def __init__(self, nmult=1., dlogmass=2., mass_kind="totpartmass",
                 overlapper_kwargs={}):
        self.nmult = nmult
        self.dlogmass = dlogmass
        self.mass_kind = mass_kind
        self._overlapper = ParticleOverlap(**overlapper_kwargs)

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

    @nmult.setter
    def nmult(self, nmult):
        """Set `nmult`."""
        assert nmult > 0
        self._nmult = nmult

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

    @dlogmass.setter
    def dlogmass(self, dlogmass):
        """Set `dlogmass`."""
        assert dlogmass > 0
        self._dlogmass = dlogmass

    @property
    def mass_kind(self):
        """
        The mass kind whose similarity is to be checked.

        Returns
        -------
        mass_kind : str
        """
        return self._mass_kind

    @mass_kind.setter
    def mass_kind(self, mass_kind):
        """Set `mass_kind`."""
        assert isinstance(mass_kind, str)
        self._mass_kind = mass_kind

    @property
    def overlapper(self):
        """
        The overlapper object.

        Returns
        -------
        overlapper : :py:class:`csiborgtools.match.ParticleOverlap`
        """
        return self._overlapper

    def cross(self, cat0, catx, clumps0, clumpsx, delta_bckg, verbose=True):
        r"""
        Find all neighbours whose CM separation is less than `nmult` times the
        sum of their initial Lagrangian patch sizes and optionally calculate
        their overlap. Enforces that the neighbours' are similar in mass up to
        `dlogmass` dex.

        Parameters
        ----------
        cat0 : :py:class:`csiborgtools.read.HaloCatalogue`
            Halo catalogue of the reference simulation.
        catx : :py:class:`csiborgtools.read.HaloCatalogue`
            Halo catalogue of the cross simulation.
        clumps0 : list of structured arrays
            List of clump structured arrays of the reference simulation, keys
            must include `x`, `y`, `z` and `M`. The positions must already be
            converted to cell numbers.
        clumpsx : list of structured arrays
            List of clump structured arrays of the cross simulation, keys must
            include `x`, `y`, `z` and `M`. The positions must already be
            converted to cell numbers.
        delta_bcgk : 3-dimensional array
            Summed background density field of the reference and cross
            simulations calculated with particles assigned to halos at the
            final snapshot. Assumed to only be sampled in cells
            :math:`[512, 1536)^3`.
        verbose : bool, optional
            iterator verbosity flag. by default `true`.

        Returns
        -------
        ref_indxs : 1-dimensional array
            Halo IDs in the reference catalogue.
        cross_indxs : 1-dimensional array
            Halo IDs in the cross catalogue.
        match_indxs : 1-dimensional array of arrays
            Indices of halo counterparts in the cross catalogue.
        overlaps : 1-dimensional array of arrays
            Overlaps with the cross catalogue.
        """
        # Query the KNN
        verbose and print("{}: querying the KNN.".format(now()), flush=True)
        match_indxs = radius_neighbours(
            catx.knn(select_initial=True), cat0.positions0,
            radiusX=cat0["lagpatch"], radiusKNN=catx["lagpatch"],
            nmult=self.nmult, enforce_in32=True, verbose=verbose)

        # Remove neighbours whose mass is too large/small
        if self.dlogmass is not None:
            for i, indx in enumerate(match_indxs):
                # |log(M1 / M2)|
                p = self.mass_kind
                aratio = numpy.abs(numpy.log10(catx[p][indx] / cat0[p][i]))
                match_indxs[i] = match_indxs[i][aratio < self.dlogmass]

        # Min and max cells along each axis for each halo
        limkwargs = {"ncells": self.overlapper.inv_clength,
                     "nshift": self.overlapper.nshift}
        mins0, maxs0 = get_clumplims(clumps0, **limkwargs)
        minsx, maxsx = get_clumplims(clumpsx, **limkwargs)

        # Mapping from a halo index to the list of clumps
        hid2clumps0 = {hid: n for n, hid in enumerate(clumps0["ID"])}
        hid2clumpsx = {hid: n for n, hid in enumerate(clumpsx["ID"])}

        cross = [None] * match_indxs.size
        # Loop only over halos that have neighbours
        iters = numpy.arange(len(cat0))[[x.size > 0 for x in match_indxs]]
        for i in tqdm(iters) if verbose else iters:
            match0 = hid2clumps0[cat0["index"][i]]
            # The clump, its mass and mins & maxs
            cl0 = clumps0["clump"][match0]
            mass0 = numpy.sum(cl0['M'])
            mins0_current, maxs0_current = mins0[match0], maxs0[match0]

            # Preallocate arrays to store overlap information
            _cross = numpy.full(match_indxs[i].size, numpy.nan,
                                dtype=numpy.float32)
            # Loop over matches of this halo from the other simulation
            for j, ind in enumerate(match_indxs[i]):
                matchx = hid2clumpsx[catx["index"][ind]]
                clx = clumpsx["clump"][matchx]
                _cross[j] = self.overlapper(
                    cl0, clx, delta_bckg, mins0_current, maxs0_current,
                    minsx[matchx], maxsx[matchx], mass1=mass0,
                    mass2=numpy.sum(clx['M']))
            cross[i] = _cross

            # Remove matches with exactly 0 overlap
            mask = cross[i] > 0
            match_indxs[i] = match_indxs[i][mask]
            cross[i] = cross[i][mask]

            # Sort the matches by overlap
            ordering = numpy.argsort(cross[i])[::-1]
            match_indxs[i] = match_indxs[i][ordering]
            cross[i] = cross[i][ordering]

        cross = numpy.asanyarray(cross, dtype=object)
        return cat0["index"], catx["index"], match_indxs, cross

    def smoothed_cross(self, clumps0, clumpsx, delta_bckg, ref_indxs,
                       cross_indxs, match_indxs, smooth_kwargs, verbose=True):
        r"""
        Calculate the smoothed overlaps for pair previously identified via
        `self.cross(...)` to have a non-zero overlap.

        Parameters
        ----------
        clumps0 : list of structured arrays
            List of clump structured arrays of the reference simulation, keys
            must include `x`, `y`, `z` and `M`. The positions must already be
            converted to cell numbers.
        clumpsx : list of structured arrays
            List of clump structured arrays of the cross simulation, keys must
            include `x`, `y`, `z` and `M`. The positions must already be
            converted to cell numbers.
        delta_bcgk : 3-dimensional array
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
        # Min and max cells along each axis for each halo
        limkwargs = {"ncells": self.overlapper.inv_clength,
                     "nshift": self.overlapper.nshift}
        mins0, maxs0 = get_clumplims(clumps0, **limkwargs)
        minsx, maxsx = get_clumplims(clumpsx, **limkwargs)

        hid2clumps0 = {hid: n for n, hid in enumerate(clumps0["ID"])}
        hid2clumpsx = {hid: n for n, hid in enumerate(clumpsx["ID"])}

        # Preallocate arrays to store the overlap information
        cross = [None] * match_indxs.size
        for i, ref_ind in enumerate(tqdm(ref_indxs) if verbose else ref_indxs):
            match0 = hid2clumps0[ref_ind]
            # The reference clump, its mass and mins & maxs
            cl0 = clumps0["clump"][match0]
            mins0_current, maxs0_current = mins0[match0], maxs0[match0]

            # Preallocate
            nmatches = match_indxs[i].size
            _cross = numpy.full(nmatches, numpy.nan, numpy.float32)
            for j, match_ind in enumerate(match_indxs[i]):
                matchx = hid2clumpsx[cross_indxs[match_ind]]
                clx = clumpsx["clump"][matchx]
                _cross[j] = self.overlapper(
                    cl0, clx, delta_bckg, mins0_current,
                    maxs0_current, minsx[matchx], maxsx[matchx],
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
    _inv_clength = None
    _clength = None
    _nshift = None

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
        return numpy.floor(pos * self.inv_clength).astype(int)

    def clumps_pos2cell(self, clumps):
        """
        Convert clump positions directly to cell IDs. Useful to speed up
        subsequent calculations. Overwrites the passed in arrays.

        Parameters
        ----------
        clumps : array of arrays
            Array of clump structured arrays whose `x`, `y`, `z` keys will be
            converted.

        Returns
        -------
        None
        """
        # Check if clumps are probably already in cells
        if any(clumps[0][0].dtype[p].char in numpy.typecodes["AllInteger"]
               for p in ('x', 'y', 'z')):
            raise ValueError("Positions appear to already be converted cells.")

        # Get the new dtype that replaces float for int for positions
        names = clumps[0][0].dtype.names  # Take the first one, doesn't matter
        formats = [descr[1] for descr in clumps[0][0].dtype.descr]

        for i in range(len(names)):
            if names[i] in ('x', 'y', 'z'):
                formats[i] = numpy.int32
        dtype = numpy.dtype({"names": names, "formats": formats})

        # Loop switch positions for cells IDs and change dtype
        for n in range(clumps.size):
            for p in ('x', 'y', 'z'):
                clumps[n][0][p] = self.pos2cell(clumps[n][0][p])
            clumps[n][0] = clumps[n][0].astype(dtype)

    def make_bckg_delta(self, clumps, delta=None):
        """
        Calculate a NGP density field of clumps within the central
        :math:`1/2^3` region of the simulation. Smoothing must be applied
        separately.

        Parameters
        ----------
        clumps : list of structured arrays
            List of clump structured array, keys must include `x`, `y`, `z`
            and `M`.
        delta : 3-dimensional array, optional
            Array to store the density field in. If `None` a new array is
            created.

        Returns
        -------
        delta : 3-dimensional array
        """
        conc_clumps = concatenate_clumps(clumps)
        cells = [self.pos2cell(conc_clumps[p]) for p in ('x', 'y', 'z')]
        mass = conc_clumps['M']

        del conc_clumps
        collect()  # This is a large array so force memory clean

        cellmin = self.inv_clength // 4         # The minimum cell ID
        cellmax = 3 * self.inv_clength // 4     # The maximum cell ID
        ncells = cellmax - cellmin
        # Mask out particles outside the cubical high resolution region
        mask = ((cellmin <= cells[0]) & (cells[0] < cellmax)
                & (cellmin <= cells[1]) & (cells[1] < cellmax)
                & (cellmin <= cells[2]) & (cells[2] < cellmax)
                )
        cells = [c[mask] for c in cells]
        mass = mass[mask]

        # Prepare the density field or check it is of the right shape
        if delta is None:
            delta = numpy.zeros((ncells,) * 3, dtype=numpy.float32)
        else:
            assert ((delta.shape == (ncells,) * 3)
                    & (delta.dtype == numpy.float32))
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

        if smooth_kwargs is not None:
            gaussian_filter(delta, output=delta, **smooth_kwargs)
        return delta

    def make_deltas(self, clump1, clump2, mins1=None, maxs1=None,
                    mins2=None, maxs2=None, smooth_kwargs=None):
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

        # If no smoothing figure out the nonzero indices of the smaller clump
        if smooth_kwargs is None:
            if clump1.size > clump2.size:
                fill_delta(delta1, xc1, yc1, zc1, *cellmins, clump1['M'])
                nonzero = fill_delta_indxs(delta2, xc2, yc2, zc2, *cellmins,
                                           clump2['M'])
            else:
                nonzero = fill_delta_indxs(delta1, xc1, yc1, zc1, *cellmins,
                                           clump1['M'])
                fill_delta(delta2, xc2, yc2, zc2, *cellmins, clump2['M'])
        else:
            fill_delta(delta1, xc1, yc1, zc1, *cellmins, clump1['M'])
            fill_delta(delta2, xc2, yc2, zc2, *cellmins, clump2['M'])
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
        delta_bcgk : 3-dimensional array
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
        mass1 = numpy.sum(clump1['M']) if mass1 is None else mass1
        mass2 = numpy.sum(clump2['M']) if mass2 is None else mass2
        return calculate_overlap_indxs(delta1, delta2, cellmins, delta_bckg,
                                       nonzero, mass1, mass2)


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
    delta_bcgk : 3-dimensional array
        Summed background density field of the reference and cross simulations
        calculated with particles assigned to halos at the final snapshot.
        Assumed to only be sampled in cells :math:`[512, 1536)^3`.

    Returns
    -------
    overlap : float
    """
    totmass = 0.           # Total mass of clump 1 and clump 2
    intersect = 0.         # Weighted intersecting mass
    i0, j0, k0 = cellmins  # Unpack things
    bckg_offset = 512      # Offset of the background density field
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
    delta_bcgk : 3-dimensional array
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
    intersect = 0.         # Weighted intersecting mass
    i0, j0, k0 = cellmins  # Unpack cell minimas
    bckg_offset = 512      # Offset of the background density field
    bckg_size = 1024       # Size of the background density field array

    for n in range(nonzero.shape[0]):
        i, j, k = nonzero[n, :]
        m1, m2 = delta1[i, j, k], delta2[i, j, k]
        prod = 2 * m1 * m2

        if prod > 0:
            ii = i0 + i - bckg_offset    # Indices of this cell in the
            jj = j0 + j - bckg_offset    # background density field.
            kk = k0 + k - bckg_offset

            ishighres = 0 <= ii < bckg_size   # Is this cell is in the high
            ishighres &= 0 <= jj < bckg_size  # resolution region for which the
            ishighres &= 0 <= kk < bckg_size  # background field is calculated.

            bckg = delta_bckg[ii, jj, kk] if ishighres else m1 + m2
            intersect += prod / bckg if bckg > 0 else prod / (m1 + m2)

    return intersect / (mass1 + mass2 - intersect)


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


def radius_neighbours(knn, X, radiusX, radiusKNN, nmult=1., enforce_in32=False,
                      verbose=True):
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
    assert X.ndim == 2 and X.shape[1] == 3       # shape of X ok?
    assert X.shape[0] == radiusX.size            # patchX matches X?
    assert radiusKNN.size == knn.n_samples_fit_  # patchknn matches the knn?

    nsamples = X.shape[0]
    indxs = [None] * nsamples
    patchknn_max = numpy.max(radiusKNN)  # Maximum for completeness

    for i in trange(nsamples) if verbose else range(nsamples):
        dist, indx = knn.radius_neighbors(X[i, :].reshape(-1, 3),
                                          radiusX[i] + patchknn_max,
                                          sort_results=True)
        # Note that `dist` and `indx` are wrapped in 1-element arrays
        # so we take the first item where appropriate
        mask = (dist[0] / (radiusX[i] + radiusKNN[indx[0]])) < nmult
        indxs[i] = indx[0][mask]
        if enforce_in32:
            indxs[i] = indxs[i].astype(numpy.int32)

    return numpy.asarray(indxs, dtype=object)
