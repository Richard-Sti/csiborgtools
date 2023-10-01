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
from tqdm import tqdm
from numba import jit


###############################################################################
#             Read in the field values at the galaxy positions                #
###############################################################################


def read_interpolated_field(survey_name, kind, galaxy_index, paths, MAS, grid,
                            in_rsp, rand_data=False, verbose=True):
    """
    Read in the interpolated field at the galaxy positions, and reorder the
    data to match the galaxy index.

    Parameters
    ----------
    survey_name : str
        Survey name.
    kind : str
        Field kind.
    galaxy_index : 1-dimensional array
        Galaxy indices to read in.
    paths : py:class:`csiborgtools.read.Paths`
        Paths manager.
    MAS : str
        Mass assignment scheme.
    grid : int
        Grid size.
    in_rsp : bool
        Whether to read in the field in redshift space.
    rand_data : bool, optional
        Whether to read in the random field data instead of the galaxy field.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    3-dimensional array of shape (nsims, len(galaxy_index), nsmooth)
    """
    nsims = paths.get_ics("csiborg")
    for i, nsim in enumerate(tqdm(nsims,
                                  desc="Reading fields",
                                  disable=not verbose)):
        fpath = paths.field_interpolated(
            survey_name, kind, MAS, grid, nsim, in_rsp=in_rsp)
        data = numpy.load(fpath)
        out_ = data["val"] if not rand_data else data["rand_val"]

        if i == 0:
            out = numpy.empty((len(nsims), *out_.shape), dtype=out_.dtype)
            indxs = data["indxs"]

        out[i] = out_

    # Reorder the data to match the survey index.
    ind2pos = {v: k for k, v in enumerate(indxs)}
    ks = numpy.empty(len(galaxy_index), dtype=numpy.int64)

    for i, k in enumerate(galaxy_index):
        j = ind2pos.get(k, None)
        if j is None:
            raise ValueError(f"There is no galaxy with index {k} in the "
                             "interpolated field.")
        ks[i] = j

    return out[:, ks, :]


###############################################################################
#            Calculate the Bayesian bootstrapped correlation                 #
###############################################################################


@jit(nopython=True, fastmath=True, boundscheck=False)
def dot_product(x, y):
    tot = 0.0
    for i in range(len(x)):
        tot += x[i] * y[i]
    return tot


@jit(nopython=True, fastmath=True, boundscheck=False)
def cov(x, y, mean_x, mean_y, weights):
    tot = 0.0
    for i in range(len(x)):
        tot += (x[i] - mean_x) * (y[i] - mean_y) * weights[i]
    return tot


@jit(nopython=True, fastmath=True, boundscheck=False)
def var(x, mean_x, weights):
    tot = 0.0
    for i in range(len(x)):
        tot += (x[i] - mean_x)**2 * weights[i]
    return tot


@jit(nopython=True, fastmath=True, boundscheck=False)
def weighted_correlation(x, y, weights):
    mean_x = dot_product(x, weights)
    mean_y = dot_product(y, weights)

    cov_xy = cov(x, y, mean_x, mean_y, weights)

    var_x = var(x, mean_x, weights)
    var_y = var(y, mean_y, weights)

    return cov_xy / numpy.sqrt(var_x * var_y)


@jit(nopython=True, fastmath=True, boundscheck=False)
def _bayesian_bootstrap_correlation(x, y, weights):
    nweights = len(weights)
    bootstrapped_correlations = numpy.full(nweights, numpy.nan, dtype=x.dtype)
    for i in range(nweights):
        bootstrapped_correlations[i] = weighted_correlation(x, y, weights[i])
    return bootstrapped_correlations


@jit(nopython=True, fastmath=True, boundscheck=False)
def rank(x):
    order = numpy.argsort(x)
    ranks = order.argsort()
    return ranks


@jit(nopython=True, fastmath=True, boundscheck=False)
def bayesian_bootstrap_correlation(x, y, kind="spearman", n_bootstrap=10000):
    """
    Calculate the Bayesian bootstrapped correlation between two arrays.

    Parameters
    ----------
    x, y : 1-dimensional arrays
        The two arrays to calculate the correlation between.
    kind : str, optional
        The type of correlation to calculate. Either `spearman` or `pearson`.
    n_bootstrap : int, optional
        The number of bootstrap samples to use.

    Returns
    -------
    corr : 1-dimensional array of shape `(n_bootstrap,)`
    """
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")

    if kind not in ["spearman", "pearson"]:
        raise ValueError("kind must be either `spearman` or `pearson`")

    if kind == "spearman":
        dtype = x.dtype
        x = rank(x).astype(dtype)
        y = rank(y).astype(dtype)

    alphas = numpy.ones(len(x), dtype=x.dtype)
    weights = numpy.random.dirichlet(alphas, size=n_bootstrap)
    return _bayesian_bootstrap_correlation(x, y, weights)


###############################################################################
#                       Distribution disagreement                             #
###############################################################################


def distribution_disagreement(x, y):
    """
    Think about this more when stacking non-Gaussian distributions.
    """
    delta = x - y
    return numpy.abs(delta.mean()) / delta.std()




"""

field will be of value (nsims, ngal, nsmooth)

Calculate the correlation for each sim and smoothing scale (nsims, nsmooth)

For each of the above stack the distributions?
"""
def correlate_at_fixed_smoothing(field_values, galaxy_property,
                                 kind="spearman", n_bootstrap=1000):
    galaxy_property = galaxy_property.astype(field_values.dtype)
    nsims = len(field_values)

    distributions = numpy.empty((nsims, n_bootstrap), dtype=field_values.dtype)

    from tqdm import trange

    for i in trange(nsims):
        distributions[i] = bayesian_bootstrap_correlation(
            field_values[i], galaxy_property, kind=kind, n_bootstrap=n_bootstrap)

    return distributions



def do_something(field_values, galaxy_property):

    pass


