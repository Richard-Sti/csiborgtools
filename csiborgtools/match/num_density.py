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
Calculation of number density functions.
"""
import numpy


def binned_counts(x, bins):
    """
    Calculate number of samples in bins.

    Parameters
    ----------
    x : 1-dimensional array
        Samples' values.
    bins : 1-dimensional array
        Bin edges of shape `(n_edges, )`.

    Returns
    -------
    centres : 1-dimensional array
        Bin centres of shape `(n_edges - 1, )`.
    counts : 1-dimensional array
        Bin counts of shape `(n_edges - 1, )`.
    """
    if not isinstance(bins, numpy.ndarray) and bins.ndim == 1:
        raise TypeError("`bins` must a 1-dimensional array.")

    n_bins = bins.size
    # Bin centres
    centres = numpy.asarray(
        [0.5 * (bins[i + 1] + bins[i]) for i in range(n_bins - 1)])
    # Bin counts
    out = numpy.full(n_bins - 1, numpy.nan, dtype=int)
    for i in range(n_bins - 1):
        out[i] = numpy.sum((x >= bins[i]) & (x < bins[i + 1]))
    return centres, out


def number_density(data, feat, bins, max_dist, to_log10, return_counts=False):
    """
    Calculate volume-limited number density of a feature `feat` from array
    `data`, normalised also by the bin width.

    Parameters
    ----------
    data : structured array
        Input array of halos.
    feat : str
        Parameter whose number density to calculate.
    bins : 1-dimensional array
        Bin edges. Note that if `to_log10` then the edges must be specified
        in the logarithmic space, not linear.
    max_dist : float
        Maximum radial distance of the volume limited sample.
    to_log10 : bool
        Whether to take a logarithm of base 10 of the feature. If so, then the
        bins must also be logarithmic.
    return_counts : bool, optional
        Whether to also return number counts in each bin. By default `False`.


    Returns
    -------
    centres : 1-dimensional array
        Bin centres of shape `(n_edges - 1, )`. If `to_log10` then converts the
        bin centres back to linear space.
    nd : 1-dimensional array
        Number density of shape `(n_edges - 1, )`.
    nd_err : 1-dimensional array
        Poissonian uncertainty of `nd` of shape `(n_edges - 1, )`.
    counts: 1-dimensional array, optional
        Counts in each bin of shape `(n_edges - 1, )`. Returned only if
        `return_counts`.
    """
    # Extract the param and optionally convert to log10
    x = data[feat]
    x = numpy.log10(x) if to_log10 else x
    # Get only things within distance from the origin
    rdist = (data["peak_x"]**2 + data["peak_y"]**2 + data["peak_z"]**2)**0.5
    x = x[rdist < max_dist]

    # Make sure bins equally spaced
    dbins = numpy.diff(bins)
    dbin = dbins[0]
    if not numpy.alltrue(dbins == dbin):
        raise ValueError("Bins must be equally spaced. Currently `{}`."
                         .format(bins))

    # Encompassed volume around the origin
    volume = 4 * numpy.pi / 3 * max_dist**3
    # Poissonian statistics
    bin_centres, counts = binned_counts(x, bins)
    nd = counts / volume / dbin
    nd_err = counts**0.5 / volume / dbin
    # Convert bins to linear space if log10
    if to_log10:
        bin_centres = 10**bin_centres

    out = (bin_centres, nd, nd_err)
    if return_counts:
        out += counts
    return out
