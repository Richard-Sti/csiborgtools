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

Right ascension and declination is always assumed to be in degrees such that
RA is in [0, 360) and dec is in [-90, 90]. Galactic coordinates are also in
degrees.
"""
from copy import deepcopy
from datetime import datetime

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from numba import jit

###############################################################################
#                           Positions                                         #
###############################################################################


@jit(nopython=True, fastmath=True, boundscheck=False)
def center_of_mass(particle_positions, particles_mass, boxsize):
    """Calculate the CM, assuming periodic boundary conditions in a cube."""
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
    Compute the 3D distance between multiple points and a reference point
    using periodic boundary conditions.
    """
    npoints = len(points)

    dist = np.zeros(npoints, dtype=points.dtype)
    for i in range(npoints):
        dist[i] = periodic_distance_two_points(
            points[i], reference_point, boxsize)

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
    """Wrap positions in a periodic box. Overwrites the input array."""
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
    Calculate the number of cells in `field` of shape `(nx, ny, nz)` that are
    non-zero.
    """
    tot = 0
    imax, jmax, kmax = field.shape
    for i in range(imax):
        for j in range(jmax):
            for k in range(kmax):
                if field[i, j, k] > 0:
                    tot += 1
    return tot


def cartesian_to_radec(X, return_degrees=True, origin=[0., 0., 0.]):
    """Calculate the radial distance, RA and deg."""
    x, y, z = X[:, 0], X[:, 1], X[:, 2]

    x -= origin[0]
    y -= origin[1]
    z -= origin[2]

    dist = np.linalg.norm(X, axis=1)
    dec = np.arcsin(z / dist)
    ra = np.arctan2(y, x)
    # Wrapping to ensure RA is in [0, 2pi) (later converted to degrees).
    ra[ra < 0] += 2 * np.pi

    if return_degrees:
        ra *= 180 / np.pi
        dec *= 180 / np.pi

    # Place the origin back
    x += origin[0]
    y += origin[1]
    z += origin[2]

    return np.vstack([dist, ra, dec]).T


def radec_to_cartesian(X):
    """
    Calculate Cartesian coordinates from radial distance, RA and dec
    `(npoints, 3)`.
    """
    dist, ra, dec = X[:, 0], X[:, 1], X[:, 2]

    cdec = np.cos(dec * np.pi / 180)
    return np.vstack([
        dist * cdec * np.cos(ra * np.pi / 180),
        dist * cdec * np.sin(ra * np.pi / 180),
        dist * np.sin(dec * np.pi / 180)
        ]).T


def radec_to_galactic(ra, dec):
    """Convert right ascension and declination to galactic coordinates."""
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    return c.galactic.l.degree, c.galactic.b.degree


@jit(nopython=True, fastmath=True, boundscheck=False)
def great_circle_distance(x1, x2, in_degrees=True):
    """
    Great circle distance between two points, each of shape `(2,)`, specified
    by RA an dec.
    """
    ra1, dec1 = x1
    ra2, dec2 = x2

    if in_degrees:
        ra1 *= np.pi / 180
        dec1 *= np.pi / 180
        ra2 *= np.pi / 180
        dec2 *= np.pi / 180

    dist = np.arccos(np.sin(dec1) * np.sin(dec2)
                     + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))

    # Convert to degrees and ensure the inputs are unchanged.
    if in_degrees:
        dist *= 180 / np.pi
        ra1 *= 180 / np.pi
        dec1 *= 180 / np.pi
        ra2 *= 180 / np.pi
        dec2 *= 180 / np.pi

    return dist


def cosine_similarity(x, y):
    r"""
    Calculate the cosine similarity between two Cartesian vectors. Defined
    as :math:`\Sum_{i} x_i y_{i} / (|x| * |y|)`. Optionally, `y` can be a
    2-dimensional array of shape `(n_samples, 3)`.
    """
    if x.ndim != 1:
        raise ValueError("`x` must be a 1-dimensional array.")

    if y.ndim == 1:
        y = y.reshape(1, -1)

    out = np.sum(x * y, axis=1)
    out /= np.linalg.norm(x) * np.linalg.norm(y, axis=1)

    return out[0] if out.size == 1 else out


def hms_to_degrees(hours, minutes=None, seconds=None):
    """Convert hours, minutes and seconds to degrees."""
    return hours * 15 + (minutes or 0) / 60 * 15 + (seconds or 0) / 3600 * 15


def dms_to_degrees(degrees, arcminutes=None, arcseconds=None):
    """Convert degrees, arcminutes and arcseconds to decimal degrees."""
    return degrees + (arcminutes or 0) / 60 + (arcseconds or 0) / 3600


def real2redshift(pos, vel, observer_location, observer_velocity, boxsize,
                  periodic_wrap=True, make_copy=True):
    r"""
    Convert real space position to redshift space position.

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


def heliocentric_to_cmb(z_helio, RA, dec, e_z_helio=None):
    """
    Convert heliocentric redshift to CMB redshift using the Planck 2018 CMB
    dipole.
    """
    # CMB dipole Planck 2018 values
    vsun_mag = 369  # km/s
    RA_sun = 167.942
    dec_sun = -6.944
    SPEED_OF_LIGHT = 299792.458  # km / s

    theta_sun = np.pi / 2 - np.deg2rad(dec_sun)
    phi_sun = np.deg2rad(RA_sun)

    # Convert to theat/phi in radians
    theta = np.pi / 2 - np.deg2rad(dec)
    phi = np.deg2rad(RA)

    # Unit vector in the direction of each galaxy
    n = np.asarray([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)]).T
    # CMB dipole unit vector
    vsun_normvect = np.asarray([np.sin(theta_sun) * np.cos(phi_sun),
                                np.sin(theta_sun) * np.sin(phi_sun),
                                np.cos(theta_sun)])

    # Project the CMB dipole onto the line of sight and normalize
    vsun_projected = vsun_mag * np.dot(n, vsun_normvect) / SPEED_OF_LIGHT

    zsun_tilde = np.sqrt((1 - vsun_projected) / (1 + vsun_projected))
    zcmb = (1 + z_helio) / zsun_tilde - 1

    # Optional linear error propagation
    if e_z_helio is not None:
        e_zcmb = np.abs(e_z_helio / zsun_tilde)
        return zcmb, e_zcmb

    return zcmb


###############################################################################
#                           Statistics                                        #
###############################################################################


@jit(nopython=True, fastmath=True, boundscheck=False)
def number_counts(x, bin_edges):
    """Calculate counts of samples in bins."""
    out = np.full(bin_edges.size - 1, np.nan, dtype=np.float32)
    for i in range(bin_edges.size - 1):
        out[i] = np.sum((x >= bin_edges[i]) & (x < bin_edges[i + 1]))
    return out


def binned_statistic(x, y, left_edges, bin_width, statistic):
    """
    Calculate a binned statistic, `statistic` must be a callable `f(x)`.
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
    package written by Collin Capano. Algorithm used is from: N. Madras and
    A.D. Sokal, J. Stat. Phys. 50, 109 (1988).
    """
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
    Thin MCMC samples (dictionary with arrays of shape `(nchains, nsamples)`)
    by the autocorrelation length of each chain and concatenate the chains.
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


def numpyro_gof(model, mcmc):
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

    Returns
    -------
    BIC, AIC: floats
    """
    samples = deepcopy(mcmc.get_samples())
    log_likelihood = samples.pop("ll_values", None)
    if log_likelihood is None:
        raise ValueError("The samples must contain the log likelihood values under the key `ll_values`.")  # noqa
    kmax = np.argmax(log_likelihood)

    # How many parameters?
    nparam = 0
    for key, val in samples.items():
        if val.ndim == 1:
            nparam += 1
        elif val.ndim == 2:
            nparam += val.shape[-1]
        else:
            raise ValueError("Invalid dimensionality of samples to count the number of parameters.")  # noqa

    try:
        ndata = model.ndata
    except AttributeError as e:
        raise AttributeError("The model must have an attribute `ndata` "
                             "indicating the number of data points.") from e

    BIC = nparam * np.log(ndata) - 2 * log_likelihood[kmax]
    AIC = 2 * nparam - 2 * log_likelihood[kmax]

    return float(BIC), float(AIC)
