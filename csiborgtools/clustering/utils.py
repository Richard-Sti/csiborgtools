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
"""Clustering support functions."""
from abc import ABC, abstractmethod
from warnings import warn

import numpy

###############################################################################
#                            Random points                                    #
###############################################################################


class BaseRVS(ABC):
    """
    Base RVS generator.
    """
    @abstractmethod
    def __call__(self, nsamples, random_state, dtype):
        """
        Generate RVS.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.
        random_state : int, optional
            Random state for the random number generator.
        dtype : numpy dtype, optional
            Data type, by default `numpy.float32`.

        Returns
        -------
        samples : 2-dimensional array of shape `(nsamples, ndim)`
        """
        pass


class RVSinsphere(BaseRVS):
    """
    Generator of uniform RVS in a sphere of radius `R` in Cartesian
    coordinates centered at the origin.

    Parameters
    ----------
    R : float
        Radius of the sphere.
    """
    def __init__(self, R):
        assert R > 0, "Radius must be positive."
        self.R = R
        BaseRVS.__init__(self)

    def __call__(self, nsamples, random_state=42, dtype=numpy.float32):
        gen = numpy.random.default_rng(random_state)
        # Spherical
        r = gen.random(nsamples, dtype=dtype)**(1/3) * self.R
        theta = 2 * numpy.arcsin(gen.random(nsamples, dtype=dtype))
        phi = 2 * numpy.pi * gen.random(nsamples, dtype=dtype)
        # Cartesian
        x = r * numpy.sin(theta) * numpy.cos(phi)
        y = r * numpy.sin(theta) * numpy.sin(phi)
        z = r * numpy.cos(theta)
        return numpy.vstack([x, y, z]).T


class RVSinbox(BaseRVS):
    r"""
    Generator of uniform RVS in a box of width `L` in Cartesian coordinates in
    :math:`[0, L]^3`.

    Parameters
    ----------
    width : float
        Width of the box.
    """
    def __init__(self, width):
        assert width > 0, "Width must be positive."
        self.width = width
        BaseRVS.__init__(self)

    def __call__(self, nsamples, random_state=42, dtype=numpy.float32):
        gen = numpy.random.default_rng(random_state)
        x = gen.random(nsamples, dtype=dtype)
        y = gen.random(nsamples, dtype=dtype)
        z = gen.random(nsamples, dtype=dtype)
        return self.width * numpy.vstack([x, y, z]).T


class RVSonsphere(BaseRVS):
    r"""
    Generator of uniform RVS on the surface of a unit sphere. RA is in
    :math:`[0, 2\pi)` and dec in :math:`[-\pi / 2, \pi / 2]`, respectively.
    If `indeg` is `True` then converted to degrees.

    Parameters
    ----------
    indeg : bool
        Whether to generate the right ascension and declination in degrees.
    """
    def __init__(self, indeg):
        assert isinstance(indeg, bool), "`indeg` must be a boolean."
        self.indeg = indeg
        BaseRVS.__init__(self)

    def __call__(self, nsamples, random_state=42, dtype=numpy.float32):
        gen = numpy.random.default_rng(random_state)
        ra = 2 * numpy.pi * gen.random(nsamples, dtype=dtype)
        dec = numpy.arcsin(2 * (gen.random(nsamples, dtype=dtype) - 0.5))
        if self.indeg:
            ra = numpy.rad2deg(ra)
            dec = numpy.rad2deg(dec)
        return numpy.vstack([ra, dec]).T


###############################################################################
#                               RA wrapping                                   #
###############################################################################


def wrapRA(ra, indeg):
    """
    Wrap RA from :math:`[-180, 180)` to :math`[0, 360)` degrees if `indeg` or
    equivalently in radians otherwise.

    Paramaters
    ----------
    ra : 1-dimensional array
        Right ascension.
    indeg : bool
        Whether the right ascension is in degrees.

    Returns
    -------
    wrapped_ra : 1-dimensional array
    """
    mask = ra < 0
    if numpy.sum(mask) == 0:
        warn("No negative right ascension found.", UserWarning, stacklevel=1)
    ra[mask] += 360 if indeg else 2 * numpy.pi
    return ra


###############################################################################
#                   Secondary assembly bias normalised marks                  #
###############################################################################


def normalised_marks(x, y, nbins):
    """
    Calculate the normalised marks of `y` binned by `x`.

    Parameters
    ----------
    x : 1-dimensional array
        Binning variable.
    y : 1-dimensional array
        The variable to be marked.
    nbins : int
        Number of percentile bins.

    Returns
    -------
    marks : 1-dimensional array
    """
    assert x.ndim == y.ndim == 1
    if y.dtype not in [numpy.float32, numpy.float64]:
        raise NotImplementedError("Marks from integers are not supported.")

    bins = numpy.percentile(x, q=numpy.linspace(0, 100, nbins + 1))
    marks = numpy.full_like(y, numpy.nan)
    for i in range(nbins):
        m = (x >= bins[i]) & (x < bins[i + 1])
        # Calculate the normalised marks of this bin
        _marks = numpy.full(numpy.sum(m), numpy.nan, dtype=marks.dtype)
        for n, ind in enumerate(numpy.argsort(y[m])):
            _marks[ind] = n
        _marks /= numpy.nanmax(_marks)
        marks[m] = _marks

    return marks
