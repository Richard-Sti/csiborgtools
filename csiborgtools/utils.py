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
Collection of stand-off utility functions used in the scripts.
"""
from copy import deepcopy
from datetime import datetime

import numpy as np
from numba import jit
from numpyro.infer import util
from scipy.stats import multivariate_normal

###############################################################################
#                           Positions                                         #
###############################################################################


@jit(nopython=True, fastmath=True, boundscheck=False)
def center_of_mass(particle_positions, particles_mass, boxsize):
    """
    Calculate the center of mass of a halo while assuming periodic boundary
    conditions of a cubical box.
    """
    cm = np.zeros(3, dtype=particle_positions.dtype)
    totmass = sum(particles_mass)

    # Convert positions to unit circle coordinates in the complex plane,
    # calculate the weighted average and convert it back to box coordinates.
    for i in range(3):
        cm_i = sum(particles_mass * np.exp(
            2j * np.pi * particle_positions[:, i] / boxsize))
        cm_i /= totmass

        cm_i = np.arctan2(cm_i.imag, cm_i.real) * boxsize / (2 * np.pi)

        if cm_i < 0:
            cm_i += boxsize
        cm[i] = cm_i

    return cm


@jit(nopython=True, fastmath=True, boundscheck=False)
def periodic_distance(points, reference_point, boxsize):
    """
    Compute the 3D distance between multiple points and a reference point using
    periodic boundary conditions.
    """
    npoints = len(points)
    half_box = boxsize / 2

    dist = np.zeros(npoints, dtype=points.dtype)
    for i in range(npoints):
        for j in range(3):
            dist_1d = abs(points[i, j] - reference_point[j])

            if dist_1d > (half_box):
                dist_1d = boxsize - dist_1d

            dist[i] += dist_1d**2

        dist[i] = dist[i]**0.5

    return dist


@jit(nopython=True, fastmath=True, boundscheck=False)
def periodic_distance_two_points(p1, p2, boxsize):
    """Compute the 3D distance between two points in a periodic box."""
    half_box = boxsize / 2

    dist = 0
    for i in range(3):
        dist_1d = abs(p1[i] - p2[i])

        if dist_1d > (half_box):
            dist_1d = boxsize - dist_1d

        dist += dist_1d**2

    return dist**0.5


@jit(nopython=True, boundscheck=False)
def periodic_wrap_grid(pos, boxsize=1):
    """Wrap positions in a periodic box."""
    for n in range(pos.shape[0]):
        for i in range(3):
            if pos[n, i] > boxsize:
                pos[n, i] -= boxsize
            elif pos[n, i] < 0:
                pos[n, i] += boxsize

    return pos


@jit(nopython=True, fastmath=True, boundscheck=False)
def delta2ncells(field):
    """
    Calculate the number of cells in `field` that are non-zero.
    """
    tot = 0
    imax, jmax, kmax = field.shape
    for i in range(imax):
        for j in range(jmax):
            for k in range(kmax):
                if field[i, j, k] > 0:
                    tot += 1
    return tot


def cartesian_to_radec(X):
    """
    Calculate the radial distance, RA [0, 360) deg and dec [-90, 90] deg.
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]

    dist = np.linalg.norm(X, axis=1)
    dec = np.arcsin(z / dist)
    ra = np.arctan2(y, x)
    ra[ra < 0] += 2 * np.pi

    ra *= 180 / np.pi
    dec *= 180 / np.pi

    return np.vstack([dist, ra, dec]).T


def radec_to_cartesian(X):
    """
    Calculate Cartesian coordinates from radial distance, RA [0, 360) deg  and
    dec [-90, 90] deg.
    """
    dist, ra, dec = X[:, 0], X[:, 1], X[:, 2]

    cdec = np.cos(dec * np.pi / 180)
    return np.vstack([
        dist * cdec * np.cos(ra * np.pi / 180),
        dist * cdec * np.sin(ra * np.pi / 180),
        dist * np.sin(dec * np.pi / 180)
        ]).T


@jit(nopython=True, fastmath=True, boundscheck=False)
def great_circle_distance(x1, x2):
    """
    Great circle distance between two points on a sphere, defined by RA and
    dec, both in degrees.
    """
    ra1, dec1 = x1
    ra2, dec2 = x2

    ra1 *= np.pi / 180
    dec1 *= np.pi / 180
    ra2 *= np.pi / 180
    dec2 *= np.pi / 180

    return 180 / np.pi * np.arccos(
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
        )


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
    """
    if x.ndim != 1:
        raise ValueError("`x` must be a 1-dimensional array.")

    if y.ndim == 1:
        y = y.reshape(1, -1)

    out = np.sum(x * y, axis=1)
    out /= np.linalg.norm(x) * np.linalg.norm(y, axis=1)

    return out[0] if out.size == 1 else out


def hms_to_degrees(hours, minutes=None, seconds=None):
    """
    Convert hours, minutes and seconds to degrees.

    Parameters
    ----------
    hours, minutes, seconds : float

    Returns
    -------
    float
    """
    return hours * 15 + (minutes or 0) / 60 * 15 + (seconds or 0) / 3600 * 15


def dms_to_degrees(degrees, arcminutes=None, arcseconds=None):
    """
    Convert degrees, arcminutes and arcseconds to decimal degrees.

    Parameters
    ----------
    degrees, arcminutes, arcseconds : float

    Returns
    -------
    float
    """
    return degrees + (arcminutes or 0) / 60 + (arcseconds or 0) / 3600


def real2redshift(pos, vel, observer_location, observer_velocity, boxsize,
                  periodic_wrap=True, make_copy=True):
    r"""
    Convert real-space position to redshift space position.

    Parameters
    ----------
    pos : 2-dimensional array `(nsamples, 3)`
        Real-space Cartesian components in `Mpc / h`.
    vel : 2-dimensional array `(nsamples, 3)`
        Cartesian velocity in `km / s`.
    observer_location: 1-dimensional array `(3,)`
        Observer location in `Mpc / h`.
    observer_velocity: 1-dimensional array `(3,)`
        Observer velocity in `km / s`.
    boxsize : float
        Box size in `Mpc / h`.
    periodic_wrap : bool, optional
        Whether to wrap around the box, particles may be outside the default
        bounds once RSD is applied.
    make_copy : bool, optional
        Whether to make a copy of `pos` before modifying it.

    Returns
    -------
    pos : 2-dimensional array `(nsamples, 3)`
        Redshift-space Cartesian position in `Mpc / h`.
    """
    if make_copy:
        pos = np.copy(pos)
        vel = np.copy(vel)

    H0_inv = 1. / 100

    # Place the observer at the origin
    pos -= observer_location
    vel -= observer_velocity

    vr_dot = np.einsum('ij,ij->i', pos, vel)
    norm2 = np.einsum('ij,ij->i', pos, pos)

    pos *= (1 + H0_inv * vr_dot / norm2).reshape(-1, 1)

    # Place the observer back
    pos += observer_location
    if not make_copy:
        vel += observer_velocity

    if periodic_wrap:
        pos = periodic_wrap_grid(pos, boxsize)

    return pos


###############################################################################
#                           Statistics                                        #
###############################################################################


@jit(nopython=True, fastmath=True, boundscheck=False)
def number_counts(x, bin_edges):
    """
    Calculate counts of samples in bins.
    """
    out = np.full(bin_edges.size - 1, np.nan, dtype=np.float32)
    for i in range(bin_edges.size - 1):
        out[i] = np.sum((x >= bin_edges[i]) & (x < bin_edges[i + 1]))
    return out


def binned_statistic(x, y, left_edges, bin_width, statistic):
    """
    Calculate a binned statistic.
    """
    out = np.full(left_edges.size, np.nan, dtype=x.dtype)

    for i in range(left_edges.size):
        mask = (x >= left_edges[i]) & (x < left_edges[i] + bin_width)

        if np.any(mask):
            out[i] = statistic(y[mask])
    return out


def fprint(msg, verbose=True):
    """Print and flush a message with a timestamp."""
    if verbose:
        print(f"{datetime.now()}:   {msg}", flush=True)


###############################################################################
#                            ACL of MCMC chains                               #
###############################################################################


def calculate_acf(data):
    """
    Calculates the autocorrelation of some data. Taken from `epsie` package
    written by Collin Capano.

    Parameters
    ----------
    data : 1-dimensional array
        The data to calculate the autocorrelation of.

    Returns
    -------
    acf : 1-dimensional array
    """
    # zero the mean
    data = data - data.mean()
    # zero-pad to 2 * nearest power of 2
    newlen = int(2**(1 + np.ceil(np.log2(len(data)))))
    x = np.zeros(newlen)
    x[:len(data)] = data[:]
    # correlate
    acf = np.correlate(x, x, mode='full')
    # drop corrupted region
    acf = acf[len(acf)//2:]
    # normalize
    acf /= acf[0]
    return acf


def calculate_acl(data):
    """
    Calculate the autocorrelation length of some data. Taken from `epsie`
    package written by Collin Capano. Algorithm used is from:
        N. Madras and A.D. Sokal, J. Stat. Phys. 50, 109 (1988).

    Parameters
    ----------
    data : 1-dimensional array
        The data to calculate the autocorrelation length of.

    Returns
    -------
    acl : int
    """
    # calculate the acf
    acf = calculate_acf(data)
    # now the ACL: Following from Sokal, this is estimated
    # as the first point where M*tau[k] <= k, where
    # tau = 2*cumsum(acf) - 1, and M is a tuneable parameter,
    # generally chosen to be = 5 (which we use here)
    m = 5
    cacf = 2. * np.cumsum(acf) - 1.
    win = m * cacf <= np.arange(len(cacf))
    if win.any():
        acl = int(np.ceil(cacf[np.where(win)[0][0]]))
    else:
        # data is too short to estimate the ACL, just choose
        # the length of the data
        acl = len(data)
    return acl


def thin_samples_by_acl(samples):
    """
    Thin MCMC samples by the autocorrelation length of each chain.

    Parameters
    ----------
    samples : dict
        Dictionary of samples. Each value is a 2-dimensional array of shape
        `(nchains, nsamples)`.

    Returns
    -------
    thinned_samples : dict
        Dictionary of thinned samples. Each value is a 1-dimensional array of
        shape `(n_thinned_samples)`, where the samples are concatenated across
        the chain.
    """
    keys = list(samples.keys())
    nchains = 1 if samples[keys[0]].ndim == 1 else samples[keys[0]].shape[0]

    samples = deepcopy(samples)

    if nchains == 1:
        for key in keys:
            samples[key] = samples[key].reshape(1, -1)

    # Calculate the ACL of each chain.
    acl = np.zeros(nchains, dtype=int)
    for i in range(nchains):
        acl[i] = max(calculate_acl(samples[key][i]) for key in keys)

    thinned_samples = {}
    for key in keys:
        key_samples = []
        for i in range(nchains):
            key_samples.append(samples[key][i, ::acl[i]])

        thinned_samples[key] = np.hstack(key_samples)

    return thinned_samples


def numpyro_gof(model, mcmc, ndata, model_kwargs={}):
    """
    Get the goodness-of-fit statistics for a sampled Numpyro model. Calculates
    the BIC and AIC using the maximum likelihood sampled point and the log
    evidence using the Laplace approximation.

    Parameters
    ----------
    model : numpyro model
        The model to evaluate.
    mcmc : numpyro MCMC
        The MCMC object containing the samples.
    ndata : int
        The number of data points.
    model_kwargs : dict, optional
        Additional keyword arguments to pass to the model.

    Returns
    -------
    gof : dict
        Dictionary containing the BIC, AIC and logZ.
    """
    samples = mcmc.get_samples(group_by_chain=False)
    log_likelihood = util.log_likelihood(model, samples, **model_kwargs)["ll"]

    # Calculate the BIC using the maximum likelihood sampled point.
    kmax = np.argmax(log_likelihood)
    nparam = len(samples)
    BIC = -2 * log_likelihood[kmax] + nparam * np.log(ndata)

    # Calculate AIC
    AIC = 2 * nparam - 2 * log_likelihood[kmax]

    # Calculate log(Z) using Laplace approximation.
    X = np.vstack([samples[key] for key in samples.keys()]).T
    mu, cov = multivariate_normal.fit(X)
    test_sample = {key: mu[i] for i, key in enumerate(samples.keys())}

    ll_mu = util.log_likelihood(model, test_sample, **model_kwargs)["ll"]
    cov_det = np.linalg.det(cov)
    D = len(mu)
    logZ = ll_mu + 0.5 * np.log(cov_det) + D / 2 * np.log(2 * np.pi)

    # Convert to float
    out = {"BIC": BIC, "AIC": AIC, "logZ": logZ}
    out = {key: float(val) for key, val in out.items()}
    return out
