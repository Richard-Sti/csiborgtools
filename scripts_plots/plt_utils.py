# Copyright (C) 2023 Richard Stiskalek
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
from scipy.stats import binned_statistic
from scipy.special import erf

dpi = 600
fout = "../plots/"
mplstyle = ["science"]


def latex_float(*floats, n=2):
    """
    Convert a float or a list of floats to a LaTeX string(s). Taken from [1].

    Parameters
    ----------
    floats : float or list of floats
        The float(s) to be converted.
    n : int, optional
        The number of significant figures to be used in the LaTeX string.

    Returns
    -------
    latex_floats : str or list of str
        The LaTeX string(s) representing the float(s).

    References
    ----------
    [1] https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python  # noqa
    """
    latex_floats = [None] * len(floats)
    for i, f in enumerate(floats):
        float_str = "{0:.{1}g}".format(f, n)
        if "e" in float_str:
            base, exponent = float_str.split("e")
            latex_floats[i] = r"{0} \times 10^{{{1}}}".format(base,
                                                              int(exponent))
        else:
            latex_floats[i] = float_str

    if len(floats) == 1:
        return latex_floats[0]
    return latex_floats


def nan_weighted_average(arr, weights=None, axis=None):
    if weights is None:
        weights = numpy.ones_like(arr)

    valid_entries = ~numpy.isnan(arr)

    # Set NaN entries in arr to 0 for computation
    arr = numpy.where(valid_entries, arr, 0)

    # Set weights of NaN entries to 0
    weights = numpy.where(valid_entries, weights, 0)

    # Compute the weighted sum and the sum of weights along the axis
    weighted_sum = numpy.sum(arr * weights, axis=axis)
    sum_weights = numpy.sum(weights, axis=axis)

    return weighted_sum / sum_weights


def nan_weighted_std(arr, weights=None, axis=None, ddof=0):
    if weights is None:
        weights = numpy.ones_like(arr)

    valid_entries = ~numpy.isnan(arr)

    # Set NaN entries in arr to 0 for computation
    arr = numpy.where(valid_entries, arr, 0)

    # Set weights of NaN entries to 0
    weights = numpy.where(valid_entries, weights, 0)

    # Calculate weighted mean
    weighted_mean = numpy.sum(
        arr * weights, axis=axis) / numpy.sum(weights, axis=axis)

    # Calculate the weighted variance
    variance = numpy.sum(
        weights * (arr - numpy.expand_dims(weighted_mean, axis))**2, axis=axis)
    variance /= numpy.sum(weights, axis=axis) - ddof

    return numpy.sqrt(variance)


def compute_error_bars(x, y, xbins, sigma):
    bin_indices = numpy.digitize(x, xbins)
    y_medians = numpy.array([numpy.median(y[bin_indices == i])
                             for i in range(1, len(xbins))])

    lower_pct = 100 * 0.5 * (1 - erf(sigma / numpy.sqrt(2)))
    upper_pct = 100 - lower_pct

    y_lower = numpy.full(len(y_medians), numpy.nan)
    y_upper = numpy.full(len(y_medians), numpy.nan)

    for i in range(len(y_medians)):
        if numpy.sum(bin_indices == i + 1) == 0:
            continue

        y_lower[i] = numpy.percentile(y[bin_indices == i + 1], lower_pct)
        y_upper[i] = numpy.percentile(y[bin_indices == i + 1], upper_pct)

    yerr = (y_medians - numpy.array(y_lower), numpy.array(y_upper) - y_medians)

    return y_medians, yerr


def normalize_hexbin(hb):
    hexagon_counts = hb.get_array()
    normalized_counts = hexagon_counts / hexagon_counts.sum()
    hb.set_array(normalized_counts)
    hb.set_clim(normalized_counts.min(), normalized_counts.max())
