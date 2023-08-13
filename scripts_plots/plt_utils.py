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


def compute_error_bars(x, y, xbins, sigma):
    bin_indices = numpy.digitize(x, xbins)
    y_medians = numpy.array([numpy.median(y[bin_indices == i])
                             for i in range(1, len(xbins))])

    lower_pct = 100 * 0.5 * (1 - erf(sigma / numpy.sqrt(2)))
    upper_pct = 100 - lower_pct

    y_lower = [numpy.percentile(y[bin_indices == i], lower_pct)
               for i in range(1, len(xbins))]
    y_upper = [numpy.percentile(y[bin_indices == i], upper_pct)
               for i in range(1, len(xbins))]

    yerr = (y_medians - numpy.array(y_lower), numpy.array(y_upper) - y_medians)

    return y_medians, yerr


def normalize_hexbin(hb):
    hexagon_counts = hb.get_array()
    normalized_counts = hexagon_counts / hexagon_counts.sum()
    hb.set_array(normalized_counts)
    hb.set_clim(normalized_counts.min(), normalized_counts.max())
