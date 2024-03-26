# Copyright (C) 2024 Richard Stiskalek
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
Validation of the CSiBORG velocity field against PV measurements. Based on [1].

References
----------
[1] https://arxiv.org/abs/1912.09383.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from warnings import catch_warnings, simplefilter, warn

import numpy as np
import numpyro
import numpyro.distributions as dist
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from h5py import File
from jax import jit
from jax import numpy as jnp
from jax import vmap
from jax.lax import cond, scan
from jax.random import PRNGKey
from numdifftools import Hessian
from numpyro.infer import Predictive, util
from scipy.interpolate import interp1d
from scipy.optimize import fmin_powell
from sklearn.model_selection import KFold
from tqdm import trange

from ..params import simname2Omega_m

SPEED_OF_LIGHT = 299792.458  # km / s


def t():
    """Shortcut to get the current time."""
    return datetime.now().strftime("%H:%M:%S")


def radec_to_galactic(ra, dec):
    """
    Convert right ascension and declination to galactic coordinates (all in
    degrees.)

    Parameters
    ----------
    ra, dec : float or 1-dimensional array
        Right ascension and declination in degrees.

    Returns
    -------
    l, b : float or 1-dimensional array
    """
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    return c.galactic.l.degree, c.galactic.b.degree


###############################################################################
#                             Data loader                                     #
###############################################################################


class DataLoader:
    """
    Data loader for the line of sight (LOS) interpolated fields and the
    corresponding catalogues.

    Parameters
    ----------
    simname : str
        Simulation name.
    ksims : int
        Index of the simulation to read in (not the IC index).
    catalogue : str
        Name of the catalogue with LOS objects.
    catalogue_fpath : str
        Path to the LOS catalogue file.
    paths : csiborgtools.read.Paths
        Paths object.
    ksmooth : int, optional
        Smoothing index.
    store_full_velocity : bool, optional
        Whether to store the full 3D velocity field. Otherwise stores only
        the radial velocity.
    """
    def __init__(self, simname, ksim, catalogue, catalogue_fpath, paths,
                 ksmooth=None, store_full_velocity=False):
        print(f"{t()}: reading the catalogue.")
        self._cat = self._read_catalogue(catalogue, catalogue_fpath)
        self._catname = catalogue

        print(f"{t()}: reading the interpolated field.")
        self._field_rdist, self._los_density, self._los_velocity = self._read_field(  # noqa
            simname, ksim, catalogue, ksmooth, paths)

        if len(self._field_rdist) % 2 == 0:
            warn(f"The number of radial steps is even. Skipping the first "
                 f"step at {self._field_rdist[0]} because Simpson's rule "
                 "requires an odd number of steps.")
            self._field_rdist = self._field_rdist[1:]
            self._los_density = self._los_density[..., 1:]
            self._los_velocity = self._los_velocity[..., 1:]

        if len(self._cat) != len(self._los_density):
            raise ValueError("The number of objects in the catalogue does not "
                             "match the number of objects in the field.")

        print(f"{t()}: calculating the radial velocity.")
        nobject = len(self._los_density)
        dtype = self._los_density.dtype

        # In case of Carrick 2015 the box is in galactic coordinates..
        if simname == "Carrick2015":
            d1, d2 = radec_to_galactic(self._cat["RA"], self._cat["DEC"])
        else:
            d1, d2 = self._cat["RA"], self._cat["DEC"]

        radvel = np.empty((nobject, len(self._field_rdist)), dtype)
        for i in range(nobject):
            radvel[i, :] = radial_velocity_los(self._los_velocity[:, i, ...],
                                               d1[i], d2[i])
        self._los_radial_velocity = radvel

        if not store_full_velocity:
            self._los_velocity = None

        self._Omega_m = simname2Omega_m(simname)

        # Normalize the CSiBORG density by the mean matter density
        if "csiborg" in simname:
            cosmo = FlatLambdaCDM(H0=100, Om0=self._Omega_m)
            mean_rho_matter = cosmo.critical_density0.to("Msun/kpc^3").value
            mean_rho_matter *= self._Omega_m
            self._los_density /= mean_rho_matter

        # Since Carrick+2015 provide `rho / <rho> - 1`
        if simname == "Carrick2015":
            self._los_density += 1

        self._mask = np.ones(len(self._cat), dtype=bool)
        self._catname = catalogue

    @property
    def cat(self):
        """
        The distance indicators catalogue.

        Returns
        -------
        structured array
        """
        return self._cat[self._mask]

    @property
    def catname(self):
        """
        Name of the catalogue.

        Returns
        -------
        str
        """
        return self._catname

    @property
    def rdist(self):
        """
        Radial distances where the field was interpolated for each object.

        Returns
        -------
        1-dimensional array
        """
        return self._field_rdist

    @property
    def los_density(self):
        """
        Density field along the line of sight.

        Returns
        ----------
        2-dimensional array of shape (n_objects, n_steps)
        """
        return self._los_density[self._mask]

    @property
    def los_velocity(self):
        """
        Velocity field along the line of sight.

        Returns
        -------
        3-dimensional array of shape (3, n_objects, n_steps)
        """
        if self._los_velocity is None:
            raise ValueError("The 3D velocities were not stored.")
        return self._los_velocity[self._mask]

    @property
    def los_radial_velocity(self):
        """
        Radial velocity along the line of sight.

        Returns
        -------
        2-dimensional array of shape (n_objects, n_steps)
        """
        return self._los_radial_velocity[self._mask]

    def _read_field(self, simname, ksim, catalogue, ksmooth, paths):
        """Read in the interpolated field."""
        nsims = paths.get_ics(simname)
        if not (0 <= ksim < len(nsims)):
            raise ValueError("Invalid simulation index.")
        nsim = nsims[ksim]

        with File(paths.field_los(simname, catalogue), 'r') as f:
            has_smoothed = True if f[f"density_{nsim}"].ndim > 2 else False
            if has_smoothed and (ksmooth is None or not isinstance(ksmooth, int)):  # noqa
                raise ValueError("The output contains smoothed field but "
                                 "`ksmooth` is None. It must be provided.")

            indx = (..., ksmooth) if has_smoothed else (...)
            los_density = f[f"density_{nsim}"][indx]
            los_velocity = f[f"velocity_{nsim}"][indx]
            rdist = f[f"rdist_{nsims[0]}"][:]

        return rdist, los_density, los_velocity

    def _read_catalogue(self, catalogue, catalogue_fpath):
        """
        Read in the distance indicator catalogue.
        """
        if catalogue == "A2":
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                arr = np.empty(len(f["RA"]), dtype=dtype)
                for key in f.keys():
                    arr[key] = f[key][:]
        elif catalogue in ["LOSS", "Foundation", "SFI_gals", "2MTF",
                           "Pantheon+"]:
            with File(catalogue_fpath, 'r') as f:
                grp = f[catalogue]

                dtype = [(key, np.float32) for key in grp.keys()]
                arr = np.empty(len(grp["RA"]), dtype=dtype)
                for key in grp.keys():
                    arr[key] = grp[key][:]
        else:
            raise ValueError(f"Unknown catalogue: `{catalogue}`.")

        return arr

    def make_jackknife_mask(self, i, n_splits, seed=42):
        """
        Set the jackknife mask to exclude the `i`-th split.

        Parameters
        ----------
        i : int
            Index of the split to exclude.
        n_splits : int
            Number of splits.
        seed : int, optional
            Random seed.

        Returns
        -------
        None, sets `mask` internally.
        """
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        n = len(self._cat)
        indxs = np.arange(n)

        gen = np.random.default_rng(seed)
        gen.shuffle(indxs)

        for j, (train_index, __) in enumerate(cv.split(np.arange(n))):
            if i == j:
                self._mask = indxs[train_index]
                return

        raise ValueError("The index `i` must be in the range of `n_splits`.")

    def reset_mask(self):
        """Reset the jackknife mask."""
        self._mask = np.ones(len(self._cat), dtype=bool)


###############################################################################
#                       Supplementary flow functions                          #
###############################################################################


def radial_velocity_los(los_velocity, ra, dec):
    """
    Calculate the radial velocity along the line of sight.

    Parameters
    ----------
    los_velocity : 2-dimensional array of shape (3, n_steps)
        Line of sight velocity field.
    ra, dec : floats
        Right ascension and declination of the line of sight.
    is_degrees : bool, optional
        Whether the angles are in degrees.

    Returns
    -------
    1-dimensional array of shape (n_steps)
    """
    types = (float, np.float32, np.float64)
    if not isinstance(ra, types) and not isinstance(dec, types):
        raise ValueError("RA and dec must be floats.")

    if los_velocity.ndim != 2 and los_velocity.shape[0] != 3:
        raise ValueError("The shape of `los_velocity` must be (3, n_steps).")

    ra_rad, dec_rad = np.deg2rad(ra), np.deg2rad(dec)

    vx, vy, vz = los_velocity
    return (vx * np.cos(ra_rad) * np.cos(dec_rad)
            + vy * np.sin(ra_rad) * np.cos(dec_rad)
            + vz * np.sin(dec_rad))


###############################################################################
#                           JAX Flow model                                    #
###############################################################################


def lognorm_mean_std_to_loc_scale(mu, std):
    """
    Calculate the location and scale parameters for the log-normal distribution
    from the mean and standard deviation.

    Parameters
    ----------
    mu, std : float
        Mean and standard deviation.

    Returns
    -------
    loc, scale : float
    """
    loc = np.log(mu) - 0.5 * np.log(1 + (std / mu) ** 2)
    scale = np.sqrt(np.log(1 + (std / mu) ** 2))
    return loc, scale


def simps(y, dx):
    """
    Simpson's rule 1D integration, assuming that the number of steps is even
    and that the step size is constant.

    Parameters
    ----------
    y : 1-dimensional array
        Function values.
    dx : float
        Step size.

    Returns
    -------
    float
    """
    if len(y) % 2 == 0:
        raise ValueError("The number of steps must be odd.")

    return dx / 3 * jnp.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])


def dist2redshift(dist, Omega_m):
    """
    Convert comoving distance to cosmological redshift if the Universe is
    flat and z << 1.

    Parameters
    ----------
    dist : float or 1-dimensional array
        Comoving distance in `Mpc / h`.
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    float or 1-dimensional array
    """
    H0 = 100
    eta = 3 * Omega_m / 2
    return 1 / eta * (1 - (1 - 2 * H0 * dist / SPEED_OF_LIGHT * eta)**0.5)


def dist2distmodulus(dist, Omega_m):
    """
    Convert comoving distance to distance modulus, assuming z << 1.

    Parameters
    ----------
    dist : float or 1-dimensional array
        Comoving distance in `Mpc / h`.
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    float or 1-dimensional array
    """
    zcosmo = dist2redshift(dist, Omega_m)
    luminosity_distance = dist * (1 + zcosmo)
    return 5 * jnp.log10(luminosity_distance) + 25


def distmodulus2dist(mu, Omega_m, ninterp=10000, zmax=0.1, mu2comoving=None,
                     return_interpolator=False):
    """
    Convert distance modulus to comoving distance. Note that this is a costly
    implementation, as it builts up the interpolator every time it is called
    unless it is provided.

    Parameters
    ----------
    mu : float or 1-dimensional array
        Distance modulus.
    Omega_m : float
        Matter density parameter.
    ninterp : int, optional
        Number of points to interpolate the mapping from distance modulus to
        comoving distance.
    zmax : float, optional
        Maximum redshift for the interpolation.
    mu2comoving : callable, optional
        Interpolator from distance modulus to comoving distance. If not
        provided, it is built up every time the function is called.
    return_interpolator : bool, optional
        Whether to return the interpolator as well.

    Returns
    -------
    float (or 1-dimensional array) and callable (optional)
    """
    if mu2comoving is None:
        zrange = np.linspace(1e-15, zmax, ninterp)
        cosmo = FlatLambdaCDM(H0=100, Om0=Omega_m)
        mu2comoving = interp1d(
            cosmo.distmod(zrange).value, cosmo.comoving_distance(zrange).value,
            kind="cubic")

    if return_interpolator:
        return mu2comoving(mu), mu2comoving

    return mu2comoving(mu)


def project_Vext(Vext_x, Vext_y, Vext_z, RA, dec):
    """
    Project the external velocity onto the line of sight along direction
    specified by RA/dec. Note that the angles must be in radians.

    Parameters
    ----------
    Vext_x, Vext_y, Vext_z : floats
        Components of the external velocity.
    RA, dec : floats
        Right ascension and declination in radians

    Returns
    -------
    float
    """
    cos_dec = jnp.cos(dec)
    return (Vext_x * jnp.cos(RA) * cos_dec
            + Vext_y * jnp.sin(RA) * cos_dec
            + Vext_z * jnp.sin(dec))


def predict_zobs(dist, beta, Vext_radial, vpec_radial, Omega_m):
    """
    Predict the observed redshift at a given comoving distance given some
    velocity field.

    Parameters
    ----------
    dist : float
        Comoving distance in `Mpc / h`.
    beta : float
        Velocity bias parameter.
    Vext_radial : float
        Radial component of the external velocity along the LOS.
    vpec_radial : float
        Radial component of the peculiar velocity along the LOS.
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    float
    """
    zcosmo = dist2redshift(dist, Omega_m)

    vrad = beta * vpec_radial + Vext_radial
    return (1 + zcosmo) * (1 + vrad / SPEED_OF_LIGHT) - 1


###############################################################################
#                          Flow validation models                             #
###############################################################################

def calculate_ptilde_wo_bias(xrange, mu, err, r_squared_xrange=None,
                             is_err_squared=False):
    """
    Calculate `ptilde(r)` without any bias.

    Parameters
    ----------
    xrange : 1-dimensional array
        Radial distances where the field was interpolated for each object.
    mu : float
        Comoving distance in `Mpc / h`.
    err : float
        Error on the comoving distance in `Mpc / h`.
    r_squared_xrange : 1-dimensional array, optional
        Radial distances squared where the field was interpolated for each
        object. If not provided, the `r^2` correction is not applied.
    is_err_squared : bool, optional
        Whether the error is already squared.

    Returns
    -------
    1-dimensional array
    """
    if is_err_squared:
        ptilde = jnp.exp(-0.5 * (xrange - mu)**2 / err)
    else:
        ptilde = jnp.exp(-0.5 * ((xrange - mu) / err)**2)

    if r_squared_xrange is not None:
        ptilde *= r_squared_xrange

    return ptilde


def calculate_ll_zobs(zobs, zobs_pred, sigma_v):
    """
    Calculate the likelihood of the observed redshift given the predicted
    redshift.

    Parameters
    ----------
    zobs : float
        Observed redshift.
    zobs_pred : float
        Predicted redshift.
    sigma_v : float
        Velocity uncertainty.

    Returns
    -------
    float
    """
    dcz = SPEED_OF_LIGHT * (zobs - zobs_pred)
    return jnp.exp(-0.5 * (dcz / sigma_v)**2) / jnp.sqrt(2 * np.pi) / sigma_v


def stack_normal(mus, stds):
    """
    Stack the normal distributions and approximate the stacked distribution
    by a single Gaussian.

    Parameters
    ----------
    mus : 1-dimensional array
        Means of the normal distributions.
    stds : 1-dimensional array
        Standard deviations of the normal distributions.

    Returns
    -------
    mu, std : floats
    """
    if mus.ndim > 1 or stds.ndim > 1 and mus.shape != stds.shape:
        raise ValueError("Shape of `mus` and `stds` must be the same and 1D.")
    mu = np.mean(mus)
    std = (np.sum(stds**2 + (mus - mu)**2) / len(mus))**0.5
    return mu, std


class BaseFlowValidationModel(ABC):
    """
    Base class for the flow validation models.
    """

    @property
    def ndata(self):
        """
        Number of PV objects in the catalogue.

        Returns
        -------
        int
        """
        return len(self._RA)

    @abstractmethod
    def predict_zobs_single(self, **kwargs):
        pass

    def predict_zobs(self, samples):
        """
        Predict the observed redshifts given the samples from the posterior.

        Parameters
        ----------
        samples : dict of 1-dimensional arrays
            Posterior samples.

        Returns
        -------
        zobs_mean : 2-dimensional array of shape (ndata, nsamples)
            Mean of the predicted redshifts.
        zobs_std : 2-dimensional array of shape (ndata, nsamples)
            Standard deviation of the predicted redshifts.
        """
        keys = list(samples.keys())
        nsamples = len(samples[keys[0]])

        zobs_mean = np.empty((self.ndata, nsamples), dtype=np.float32)
        zobs_std = np.empty_like(zobs_mean)

        # JIT compile the function, it is called many times.
        f = jit(self.predict_zobs_single)

        for i in trange(nsamples):
            x = {key: samples[key][i] for key in keys}
            if "alpha" not in keys:
                x["alpha"] = 1.0

            e_z = samples["sigma_v"][i] / SPEED_OF_LIGHT

            mu, var = f(**x)
            zobs_mean[:, i] = mu
            zobs_std[:, i] = (var + e_z**2)**0.5

        return zobs_mean, zobs_std

    def summarize_zobs_pred(self, zobs_mean, zobs_pred):
        """
        Summarize the predicted observed redshifts from each posterior sample
        by stacking their Gaussian distribution and approximating the stacked
        distribution by a single Gaussian.

        Parameters
        ----------
        zobs_mean : 2-dimensional array of shape (ndata, nsamples)
            Mean of the predicted redshifts.
        zobs_pred : 2-dimensional array of shape (ndata, nsamples)
            Predicted redshifts.

        Returns
        -------
        mu : 1-dimensional array
            Mean of predicted redshift, averaged over the posterior samples.
        std : 1-dimensional array
            Standard deviation of the predicted redshift, averaged over the
            posterior samples.
        """
        mu = np.empty(self.ndata, dtype=np.float32)
        std = np.empty_like(mu)

        for i in range(self.ndata):
            mu[i], std[i] = stack_normal(zobs_mean[i], zobs_pred[i])

        return mu, std

    @abstractmethod
    def __call__(self, **kwargs):
        pass


class SD_PV_validation_model(BaseFlowValidationModel):
    """
    Simple distance peculiar velocity (PV) validation model, assuming that
    we already have a calibrated estimate of the comoving distance to the
    objects.

    Parameters
    ----------
    los_density : 2-dimensional array of shape (n_objects, n_steps)
        LOS density field.
    los_velocity : 3-dimensional array of shape (n_objects, n_steps)
        LOS radial velocity field.
    RA, dec : 1-dimensional arrays of shape (n_objects)
        Right ascension and declination in degrees.
    z_obs : 1-dimensional array of shape (n_objects)
        Observed redshifts.
    r_hMpc : 1-dimensional array of shape (n_objects)
        Estimated comoving distances in `h^-1 Mpc`.
    e_r_hMpc : 1-dimensional array of shape (n_objects)
        Errors on the estimated comoving distances in `h^-1 Mpc`.
    r_xrange : 1-dimensional array
        Radial distances where the field was interpolated for each object.
    Omega_m : float
        Matter density parameter.
    """

    def __init__(self, los_density, los_velocity, RA, dec, z_obs,
                 r_hMpc, e_r_hMpc, r_xrange, Omega_m):
        dt = jnp.float32
        # Convert everything to JAX arrays.
        self._los_density = jnp.asarray(los_density, dtype=dt)
        self._los_velocity = jnp.asarray(los_velocity, dtype=dt)

        self._RA = jnp.asarray(np.deg2rad(RA), dtype=dt)
        self._dec = jnp.asarray(np.deg2rad(dec), dtype=dt)
        self._z_obs = jnp.asarray(z_obs, dtype=dt)

        self._r_hMpc = jnp.asarray(r_hMpc, dtype=dt)
        self._e2_rhMpc = jnp.asarray(e_r_hMpc**2, dtype=dt)

        # Get radius squared
        r2_xrange = r_xrange**2
        r2_xrange /= r2_xrange.mean()

        # Get the stepsize, we need it to be constant for Simpson's rule.
        dr = np.diff(r_xrange)
        if not np.all(np.isclose(dr, dr[0], atol=1e-5)):
            raise ValueError("The radial step size must be constant.")
        dr = dr[0]

        self._r_xrange = r_xrange

        # Get the various vmapped functions
        self._vmap_ptilde_wo_bias = vmap(lambda mu, err: calculate_ptilde_wo_bias(r_xrange, mu, err, r2_xrange, True))                  # noqa
        self._vmap_simps = vmap(lambda y: simps(y, dr))
        self._vmap_zobs = vmap(lambda beta, Vr, vpec_rad: predict_zobs(r_xrange, beta, Vr, vpec_rad, Omega_m), in_axes=(None, 0, 0))    # noqa
        self._vmap_ll_zobs = vmap(lambda zobs, zobs_pred, sigma_v: calculate_ll_zobs(zobs, zobs_pred, sigma_v), in_axes=(0, 0, None))   # noqa

        # Distribution of external velocity components
        self._Vext = dist.Uniform(-500, 500)
        # Distribution of density, velocity and location bias parameters
        self._alpha = dist.LogNormal(*lognorm_mean_std_to_loc_scale(1.0, 0.5))     # noqa
        self._beta = dist.Normal(1., 0.5)
        # Distribution of velocity uncertainty sigma_v
        self._sv = dist.LogNormal(*lognorm_mean_std_to_loc_scale(150, 100))

    def predict_zobs_single(self, **kwargs):
        raise NotImplementedError("This method is not implemented yet.")

    def __call__(self, sample_alpha=False):
        """
        The simple distance NumPyro PV validation model.

        Parameters
        ----------
        sample_alpha : bool, optional
            Whether to sample the density bias parameter `alpha`, otherwise
            it is fixed to 1.
        """
        Vx = numpyro.sample("Vext_x", self._Vext)
        Vy = numpyro.sample("Vext_y", self._Vext)
        Vz = numpyro.sample("Vext_z", self._Vext)

        alpha = numpyro.sample("alpha", self._alpha) if sample_alpha else 1.0
        beta = numpyro.sample("beta", self._beta)
        sigma_v = numpyro.sample("sigma_v", self._sv)

        Vext_rad = project_Vext(Vx, Vy, Vz, self._RA, self._dec)

        # Calculate p(r) and multiply it by the galaxy bias
        ptilde = self._vmap_ptilde_wo_bias(self._r_hMpc, self._e2_rhMpc)
        ptilde *= self._los_density**alpha

        # Normalization of p(r)
        pnorm = self._vmap_simps(ptilde)

        # Calculate p(z_obs) and multiply it by p(r)
        zobs_pred = self._vmap_zobs(beta, Vext_rad, self._los_velocity)
        ptilde *= self._vmap_ll_zobs(self._z_obs, zobs_pred, sigma_v)

        ll = jnp.sum(jnp.log(self._vmap_simps(ptilde) / pnorm))
        numpyro.factor("ll", ll)


class SN_PV_validation_model(BaseFlowValidationModel):
    """
    Supernova peculiar velocity (PV) validation model that includes the
    calibration of the SALT2 light curve parameters.

    Parameters
    ----------
    los_density : 2-dimensional array of shape (n_objects, n_steps)
        LOS density field.
    los_velocity : 3-dimensional array of shape (n_objects, n_steps)
        LOS radial velocity field.
    RA, dec : 1-dimensional arrays of shape (n_objects)
        Right ascension and declination in degrees.
    z_obs : 1-dimensional array of shape (n_objects)
        Observed redshifts.
    mB, x1, c : 1-dimensional arrays of shape (n_objects)
        SALT2 light curve parameters.
    e_mB, e_x1, e_c : 1-dimensional arrays of shape (n_objects)
        Errors on the SALT2 light curve parameters.
    r_xrange : 1-dimensional array
        Radial distances where the field was interpolated for each object.
    Omega_m : float
        Matter density parameter.
    """

    def __init__(self, los_density, los_velocity, RA, dec, z_obs,
                 mB, x1, c, e_mB, e_x1, e_c, r_xrange, Omega_m):
        dt = jnp.float32
        # Convert everything to JAX arrays.
        self._los_density = jnp.asarray(los_density, dtype=dt)
        self._los_velocity = jnp.asarray(los_velocity, dtype=dt)

        self._RA = jnp.asarray(np.deg2rad(RA), dtype=dt)
        self._dec = jnp.asarray(np.deg2rad(dec), dtype=dt)
        self._z_obs = jnp.asarray(z_obs, dtype=dt)

        self._mB = jnp.asarray(mB, dtype=dt)
        self._x1 = jnp.asarray(x1, dtype=dt)
        self._c = jnp.asarray(c, dtype=dt)
        self._e2_mB = jnp.asarray(e_mB**2, dtype=dt)
        self._e2_x1 = jnp.asarray(e_x1**2, dtype=dt)
        self._e2_c = jnp.asarray(e_c**2, dtype=dt)

        # Get radius squared
        r2_xrange = r_xrange**2
        r2_xrange /= r2_xrange.mean()
        mu_xrange = dist2distmodulus(r_xrange, Omega_m)

        # Get the stepsize, we need it to be constant for Simpson's rule.
        dr = np.diff(r_xrange)
        if not np.all(np.isclose(dr, dr[0], atol=1e-5)):
            raise ValueError("The radial step size must be constant.")
        dr = dr[0]

        # Get the various functions, also vmapped
        self._f_ptilde_wo_bias = lambda mu, err: calculate_ptilde_wo_bias(mu_xrange, mu, err, r2_xrange, True)  # noqa
        self._f_simps = lambda y: simps(y, dr)                                                                  # noqa
        self._f_zobs = lambda beta, Vr, vpec_rad: predict_zobs(r_xrange, beta, Vr, vpec_rad, Omega_m)           # noqa

        self._vmap_ptilde_wo_bias = vmap(lambda mu, err: calculate_ptilde_wo_bias(mu_xrange, mu, err, r2_xrange, True))                  # noqa
        self._vmap_simps = vmap(lambda y: simps(y, dr))
        self._vmap_zobs = vmap(lambda beta, Vr, vpec_rad: predict_zobs(r_xrange, beta, Vr, vpec_rad, Omega_m), in_axes=(None, 0, 0))    # noqa

        # Distribution of external velocity components
        self._Vext = dist.Uniform(-500, 500)
        # Distribution of velocity and density bias parameters
        self._alpha = dist.LogNormal(*lognorm_mean_std_to_loc_scale(1.0, 0.5))
        self._beta = dist.Normal(1., 0.5)
        # Distribution of velocity uncertainty
        self._sigma_v = dist.LogNormal(*lognorm_mean_std_to_loc_scale(150, 100))   # noqa

        # Distribution of light curve calibration parameters
        self._mag_cal = dist.Normal(-18.25, 0.5)
        self._alpha_cal = dist.Normal(0.148, 0.05)
        self._beta_cal = dist.Normal(3.112, 1.0)
        self._e_mu = dist.LogNormal(*lognorm_mean_std_to_loc_scale(0.1, 0.05))

        self._Omega_m = Omega_m
        self._r_xrange = r_xrange

    def mu(self, mag_cal, alpha_cal, beta_cal):
        """
        Distance modulus of each object the given SALT2 calibration parameters.

        Parameters
        ----------
        mag_cal, alpha_cal, beta_cal : floats
            SALT2 calibration parameters.

        Returns
        -------
        1-dimensional array
        """
        return self._mB - mag_cal + alpha_cal * self._x1 - beta_cal * self._c

    def squared_e_mu(self, alpha_cal, beta_cal, e_mu_intrinsic):
        """
        Linearly-propagated squared error on the SALT2 distance modulus.

        Parameters
        ----------
        alpha_cal, beta_cal, e_mu_intrinsic : floats
            SALT2 calibration parameters.

        Returns
        -------
        1-dimensional array
        """
        return (self._e2_mB + alpha_cal**2 * self._e2_x1
                + beta_cal**2 * self._e2_c + e_mu_intrinsic**2)

    def predict_zobs_single(self, Vext_x, Vext_y, Vext_z, alpha, beta,
                            e_mu_intrinsic, mag_cal, alpha_cal, beta_cal,
                            **kwargs):
        """
        Predict the observed redshifts given the samples from the posterior.

        Parameters
        ----------
        Vext_x, Vext_y, Vext_z : floats
            Components of the external velocity.
        alpha, beta : floats
            Density and velocity bias parameters.
        e_mu_intrinsic, mag_cal, alpha_cal, beta_cal : floats
            Calibration parameters.
        kwargs : dict
            Additional arguments (for compatibility).

        Returns
        -------
        zobs_mean : 1-dimensional array
            Mean of the predicted redshifts.
        zobs_var : 1-dimensional array
            Variance of the predicted redshifts.
        """
        mu = self.mu(mag_cal, alpha_cal, beta_cal)
        squared_e_mu = self.squared_e_mu(alpha_cal, beta_cal, e_mu_intrinsic)
        Vext_rad = project_Vext(Vext_x, Vext_y, Vext_z, self._RA, self._dec)

        # Calculate p(r) (Malmquist bias)
        ptilde = self._vmap_ptilde_wo_bias(mu, squared_e_mu)
        ptilde *= self._los_density**alpha
        ptilde /= self._vmap_simps(ptilde).reshape(-1, 1)

        # Predicted mean z_obs
        zobs_pred = self._vmap_zobs(beta, Vext_rad, self._los_velocity)
        zobs_mean = self._vmap_simps(zobs_pred * ptilde)

        # Variance of the predicted z_obs
        zobs_pred -= zobs_mean.reshape(-1, 1)
        zobs_var = self._vmap_simps(zobs_pred**2 * ptilde)

        return zobs_mean, zobs_var

    def __call__(self, sample_alpha=True, sample_beta=True):
        """
        The supernova NumPyro PV validation model with SALT2 calibration.

        Parameters
        ----------
        sample_alpha : bool, optional
            Whether to sample the density bias parameter `alpha`, otherwise
            it is fixed to 1.
        sample_beta : bool, optional
            Whether to sample the velocity bias parameter `beta`, otherwise
            it is fixed to 1.

        Returns
        -------
        None
        """
        Vx = numpyro.sample("Vext_x", self._Vext)
        Vy = numpyro.sample("Vext_y", self._Vext)
        Vz = numpyro.sample("Vext_z", self._Vext)
        alpha = numpyro.sample("alpha", self._alpha) if sample_alpha else 1.0
        beta = numpyro.sample("beta", self._beta) if sample_beta else 1.0
        sigma_v = numpyro.sample("sigma_v", self._sigma_v)

        e_mu_intrinsic = numpyro.sample("e_mu_intrinsic", self._e_mu)
        mag_cal = numpyro.sample("mag_cal", self._mag_cal)
        alpha_cal = numpyro.sample("alpha_cal", self._alpha_cal)
        beta_cal = numpyro.sample("beta_cal", self._beta_cal)

        Vext_rad = project_Vext(Vx, Vy, Vz, self._RA, self._dec)

        mu = self.mu(mag_cal, alpha_cal, beta_cal)
        squared_e_mu = self.squared_e_mu(alpha_cal, beta_cal, e_mu_intrinsic)

        def scan_body(ll, i):
            # Calculate p(r) and multiply it by the galaxy bias
            ptilde = self._f_ptilde_wo_bias(mu[i], squared_e_mu[i])
            ptilde *= self._los_density[i]**alpha

            # Normalization of p(r)
            pnorm = self._f_simps(ptilde)

            # Calculate p(z_obs) and multiply it by p(r)
            zobs_pred = self._f_zobs(beta, Vext_rad[i], self._los_velocity[i])
            ptilde *= calculate_ll_zobs(self._z_obs[i], zobs_pred, sigma_v)

            return ll + jnp.log(self._f_simps(ptilde) / pnorm), None

        ll = 0.
        ll, __ = scan(scan_body, ll, jnp.arange(self.ndata))
        numpyro.factor("ll", ll)


class TF_PV_validation_model(BaseFlowValidationModel):
    """
    Tully-Fisher peculiar velocity (PV) validation model that includes the
    calibration of the Tully-Fisher distance `mu = m - (a + b * eta)`.

    Parameters
    ----------
    los_density : 2-dimensional array of shape (n_objects, n_steps)
        LOS density field.
    los_velocity : 3-dimensional array of shape (n_objects, n_steps)
        LOS radial velocity field.
    RA, dec : 1-dimensional arrays of shape (n_objects)
        Right ascension and declination in degrees.
    z_obs : 1-dimensional array of shape (n_objects)
        Observed redshifts.
    mag, eta : 1-dimensional arrays of shape (n_objects)
        Apparent magnitude and `eta` parameter.
    e_mag, e_eta : 1-dimensional arrays of shape (n_objects)
        Errors on the apparent magnitude and `eta` parameter.
    r_xrange : 1-dimensional array
        Radial distances where the field was interpolated for each object.
    Omega_m : float
        Matter density parameter.
    """

    def __init__(self, los_density, los_velocity, RA, dec, z_obs,
                 mag, eta, e_mag, e_eta, r_xrange, Omega_m):
        dt = jnp.float32
        # Convert everything to JAX arrays.
        self._los_density = jnp.asarray(los_density, dtype=dt)
        self._los_velocity = jnp.asarray(los_velocity, dtype=dt)

        self._RA = jnp.asarray(np.deg2rad(RA), dtype=dt)
        self._dec = jnp.asarray(np.deg2rad(dec), dtype=dt)
        self._z_obs = jnp.asarray(z_obs, dtype=dt)

        self._mag = jnp.asarray(mag, dtype=dt)
        self._eta = jnp.asarray(eta, dtype=dt)
        self._e2_mag = jnp.asarray(e_mag**2, dtype=dt)
        self._e2_eta = jnp.asarray(e_eta**2, dtype=dt)

        # Get radius squared
        r2_xrange = r_xrange**2
        r2_xrange /= r2_xrange.mean()
        mu_xrange = dist2distmodulus(r_xrange, Omega_m)

        # Get the stepsize, we need it to be constant for Simpson's rule.
        dr = np.diff(r_xrange)
        if not np.all(np.isclose(dr, dr[0], atol=1e-5)):
            raise ValueError("The radial step size must be constant.")
        dr = dr[0]

        # Get the various vmapped functions
        self._f_ptilde_wo_bias = lambda mu, err: calculate_ptilde_wo_bias(mu_xrange, mu, err, r2_xrange, True)  # noqa
        self._f_simps = lambda y: simps(y, dr)                                                                  # noqa
        self._f_zobs = lambda beta, Vr, vpec_rad: predict_zobs(r_xrange, beta, Vr, vpec_rad, Omega_m)           # noqa

        self._vmap_ptilde_wo_bias = vmap(lambda mu, err: calculate_ptilde_wo_bias(mu_xrange, mu, err, r2_xrange, True))                  # noqa
        self._vmap_simps = vmap(lambda y: simps(y, dr))
        self._vmap_zobs = vmap(lambda beta, Vr, vpec_rad: predict_zobs(r_xrange, beta, Vr, vpec_rad, Omega_m), in_axes=(None, 0, 0))    # noqa

        # Distribution of external velocity components
        self._Vext = dist.Uniform(-1000, 1000)
        # Distribution of velocity and density bias parameters
        self._alpha = dist.LogNormal(*lognorm_mean_std_to_loc_scale(1.0, 0.5))     # noqa
        self._beta = dist.Normal(1., 0.5)
        # Distribution of velocity uncertainty
        self._sigma_v = dist.LogNormal(*lognorm_mean_std_to_loc_scale(150, 100))   # noqa

        # Distribution of Tully-Fisher calibration parameters
        self._a = dist.Normal(-21., 0.5)
        self._b = dist.Normal(-5.95, 0.1)
        self._e_mu = dist.LogNormal(*lognorm_mean_std_to_loc_scale(0.3, 0.1))      # noqa

        self._Omega_m = Omega_m
        self._r_xrange = r_xrange

    def mu(self, a, b):
        """
        Distance modulus of each object the given Tully-Fisher calibration.

        Parameters
        ----------
        a, b : floats
            Tully-Fisher calibration parameters.

        Returns
        -------
        1-dimensional array
        """

        return self._mag - (a + b * self._eta)

    def squared_e_mu(self, b, e_mu_intrinsic):
        """
        Squared error on the Tully-Fisher distance modulus.

        Parameters
        ----------
        b, e_mu_intrinsic : floats
            Tully-Fisher calibration parameters.

        Returns
        -------
        1-dimensional array
        """
        return (self._e2_mag + b**2 * self._e2_eta + e_mu_intrinsic**2)

    def predict_zobs_single(self, Vext_x, Vext_y, Vext_z, alpha, beta,
                            e_mu_intrinsic, a, b, **kwargs):
        """
        Predict the observed redshifts given the samples from the posterior.

        Parameters
        ----------
        Vext_x, Vext_y, Vext_z : floats
            Components of the external velocity.
        alpha, beta : floats
            Density and velocity bias parameters.
        e_mu_intrinsic, a, b : floats
            Calibration parameters.
        kwargs : dict
            Additional arguments (for compatibility).

        Returns
        -------
        zobs_mean : 1-dimensional array
            Mean of the predicted redshifts.
        zobs_var : 1-dimensional array
            Variance of the predicted redshifts.
        """
        mu = self.mu(a, b)
        squared_e_mu = self.squared_e_mu(b, e_mu_intrinsic)

        Vext_rad = project_Vext(Vext_x, Vext_y, Vext_z, self._RA, self._dec)

        # Calculate p(r) (Malmquist bias)
        ptilde = self._vmap_ptilde_wo_bias(mu, squared_e_mu)
        ptilde *= self._los_density**alpha
        ptilde /= self._vmap_simps(ptilde).reshape(-1, 1)

        # Predicted mean z_obs
        zobs_pred = self._vmap_zobs(beta, Vext_rad, self._los_velocity)
        zobs_mean = self._vmap_simps(zobs_pred * ptilde)

        # Variance of the predicted z_obs
        zobs_pred -= zobs_mean.reshape(-1, 1)
        zobs_var = self._vmap_simps(zobs_pred**2 * ptilde)

        return zobs_mean, zobs_var

    def __call__(self, sample_alpha=True, sample_beta=True):
        """
        The Tully-Fisher NumPyro PV validation model.

        Parameters
        ----------
        sample_alpha : bool, optional
            Whether to sample the density bias parameter `alpha`, otherwise
            it is fixed to 1.
        sample_beta : bool, optional
            Whether to sample the velocity bias parameter `beta`, otherwise
            it is fixed to 1.

        Returns
        -------
        None
        """
        Vx = numpyro.sample("Vext_x", self._Vext)
        Vy = numpyro.sample("Vext_y", self._Vext)
        Vz = numpyro.sample("Vext_z", self._Vext)
        alpha = numpyro.sample("alpha", self._alpha) if sample_alpha else 1.0
        beta = numpyro.sample("beta", self._beta) if sample_beta else 1.0
        sigma_v = numpyro.sample("sigma_v", self._sigma_v)

        e_mu_intrinsic = numpyro.sample("e_mu_intrinsic", self._e_mu)
        a = numpyro.sample("a", self._a)
        b = numpyro.sample("b", self._b)

        Vext_rad = project_Vext(Vx, Vy, Vz, self._RA, self._dec)

        mu = self.mu(a, b)
        squared_e_mu = self.squared_e_mu(b, e_mu_intrinsic)

        def scan_body(ll, i):
            # Calculate p(r) and multiply it by the galaxy bias
            ptilde = self._f_ptilde_wo_bias(mu[i], squared_e_mu[i])
            ptilde *= self._los_density[i]**alpha

            # Normalization of p(r)
            pnorm = self._f_simps(ptilde)

            # Calculate p(z_obs) and multiply it by p(r)
            zobs_pred = self._f_zobs(beta, Vext_rad[i], self._los_velocity[i])
            ptilde *= calculate_ll_zobs(self._z_obs[i], zobs_pred, sigma_v)

            return ll + jnp.log(self._f_simps(ptilde) / pnorm), None

        ll = 0.
        ll, __ = scan(scan_body, ll, jnp.arange(self.ndata))

        numpyro.factor("ll", ll)


###############################################################################
#                       Shortcut to create a model                           #
###############################################################################


def get_model(loader, zcmb_max=None, verbose=True):
    """
    Get a model and extract the relevant data from the loader.

    Parameters
    ----------
    loader : DataLoader
        DataLoader instance.
    zcmb_max : float, optional
        Maximum observed redshift in the CMB frame to include.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    model : NumPyro model
    """
    zcmb_max = np.infty if zcmb_max is None else zcmb_max

    los_overdensity = loader.los_density
    los_velocity = loader.los_radial_velocity
    kind = loader._catname

    if kind in ["LOSS", "Foundation"]:
        keys = ["RA", "DEC", "z_CMB", "mB", "x1", "c", "e_mB", "e_x1", "e_c"]
        RA, dec, zCMB, mB, x1, c, e_mB, e_x1, e_c = (loader.cat[k] for k in keys)  # noqa

        mask = (zCMB < zcmb_max)
        model = SN_PV_validation_model(
            los_overdensity[mask], los_velocity[mask], RA[mask], dec[mask],
            zCMB[mask], mB[mask], x1[mask], c[mask], e_mB[mask], e_x1[mask],
            e_c[mask], loader.rdist, loader._Omega_m)
    elif kind == "Pantheon+":
        keys = ["RA", "DEC", "zCMB", "mB", "x1", "c", "biasCor_m_b", "mBERR",
                "x1ERR", "cERR", "biasCorErr_m_b"]

        RA, dec, zCMB, mB, x1, c, bias_corr_mB, e_mB, e_x1, e_c, e_bias_corr_mB = (loader.cat[k] for k in keys)  # noqa
        mB -= bias_corr_mB
        e_mB = np.sqrt(e_mB**2 + e_bias_corr_mB**2)

        mask = (zCMB < zcmb_max)
        model = SN_PV_validation_model(
            los_overdensity[mask], los_velocity[mask], RA[mask], dec[mask],
            zCMB[mask], mB[mask], x1[mask], c[mask], e_mB[mask], e_x1[mask],
            e_c[mask], loader.rdist, loader._Omega_m)
    elif kind in ["SFI_gals", "2MTF"]:
        keys = ["RA", "DEC", "z_CMB", "mag", "eta", "e_mag", "e_eta"]
        RA, dec, zCMB, mag, eta, e_mag, e_eta = (loader.cat[k] for k in keys)

        mask = (zCMB < zcmb_max)
        if kind == "SFI_gals":
            mask &= (eta > -0.15) & (eta < 0.2)
            if verbose:
                print("Emplyed eta cut for SFI galaxies.", flush=True)
        model = TF_PV_validation_model(
            los_overdensity[mask], los_velocity[mask], RA[mask], dec[mask],
            zCMB[mask], mag[mask], eta[mask], e_mag[mask], e_eta[mask],
            loader.rdist, loader._Omega_m)
    else:
        raise ValueError(f"Catalogue `{kind}` not recognized.")

    if verbose:
        print(f"Selected {np.sum(mask)}/{len(mask)} galaxies.", flush=True)

    return model


###############################################################################
#                  Maximizing likelihood of a NumPyro model                   #
###############################################################################


def sample_prior(model, seed, model_kwargs, as_dict=False):
    """
    Sample a single set of parameters from the prior of the model.

    Parameters
    ----------
    model : NumPyro model
        NumPyro model.
    seed : int
        Random seed.
    model_kwargs : dict
        Additional keyword arguments to pass to the model.
    as_dict : bool, optional
        Whether to return the parameters as a dictionary or a list of
        parameters.

    Returns
    -------
    x, keys : tuple
        Tuple of parameters and their names. If `as_dict` is True, returns
        only a dictionary.
    """
    predictive = Predictive(model, num_samples=1)
    samples = predictive(PRNGKey(seed), **model_kwargs)

    if as_dict:
        return samples

    keys = list(samples.keys())
    if "ll" in keys:
        keys.remove("ll")

    x = np.asarray([samples[key][0] for key in keys])
    return x, keys


def make_loss(model, keys, model_kwargs, to_jit=True):
    """
    Generate a loss function for the NumPyro model, that is the negative
    log-likelihood. Note that this loss function cannot be automatically
    differentiated.

    Parameters
    ----------
    model : NumPyro model
        NumPyro model.
    keys : list
        List of parameter names.
    model_kwargs : dict
        Additional keyword arguments to pass to the model.
    to_jit : bool, optional
        Whether to JIT the loss function.

    Returns
    -------
    loss : function
        Loss function `f(x)` where `x` is a list of parameters ordered
        according to `keys`.
    """
    def f(x):
        samples = {key: x[i] for i, key in enumerate(keys)}

        loss = -util.log_likelihood(model, samples, **model_kwargs)["ll"]

        loss += cond(samples["sigma_v"] > 0, lambda: 0., lambda: jnp.inf)
        loss += cond(samples["e_mu_intrinsic"] > 0, lambda: 0., lambda: jnp.inf)  # noqa

        return cond(jnp.isfinite(loss), lambda: loss, lambda: jnp.inf)

    if to_jit:
        return jit(f)

    return f


def optimize_model_with_jackknife(loader, k, n_splits=5, sample_alpha=True,
                                  get_model_kwargs={}, seed=42):
    """
    Optimize the log-likelihood of a model for `n_splits` jackknifes.

    Parameters
    ----------
    loader : DataLoader
        DataLoader instance.
    k : int
        Simulation index.
    n_splits : int, optional
        Number of jackknife splits.
    sample_alpha : bool, optional
        Whether to sample the density bias parameter `alpha`.
    get_model_kwargs : dict, optional
        Additional keyword arguments to pass to the `get_model` function.
    seed : int, optional
        Random seed.

    Returns
    -------
    samples : dict
        Dictionary of optimized parameters for each jackknife split.
    stats : dict
        Dictionary of mean and standard deviation for each parameter.
    fmin : 1-dimensional array
        Minimum negative log-likelihood for each jackknife split.
    logz : 1-dimensional array
        Log-evidence for each jackknife split.
    bic : 1-dimensional array
        Bayesian information criterion for each jackknife split.
    """
    mask = np.zeros(n_splits, dtype=bool)
    x0 = None

    # Loop over the CV splits.
    for i in trange(n_splits):
        loader.make_jackknife_mask(i, n_splits, seed=seed)
        model = get_model(loader, k, verbose=False, **get_model_kwargs)

        if x0 is None:
            x0, keys = sample_prior(model, seed, sample_alpha)
            x = np.full((n_splits, len(x0)), np.nan)
            fmin = np.full(n_splits, np.nan)
            logz = np.full(n_splits, np.nan)
            bic = np.full(n_splits, np.nan)

            loss = make_loss(model, keys, sample_alpha=sample_alpha,
                             to_jit=True)
            for j in range(100):
                if np.isfinite(loss(x0)):
                    break
                x0, __ = sample_prior(model, seed + 1, sample_alpha)
            else:
                raise ValueError("Failed to find finite initial loss.")

        else:
            loss = make_loss(model, keys, sample_alpha=sample_alpha,
                             to_jit=True)

        with catch_warnings():
            simplefilter("ignore")
            res = fmin_powell(loss, x0, disp=False)

        if np.all(np.isfinite(res)):
            x[i] = res
            mask[i] = True
            x0 = res
            fmin[i] = loss(res)

            f_hess = Hessian(loss, method="forward", richardson_terms=1)
            hess = f_hess(res)
            D = len(keys)
            logz[i] = (
                - fmin[i]
                + 0.5 * np.log(np.abs(np.linalg.det(np.linalg.inv(hess))))
                + D / 2 * np.log(2 * np.pi))

            bic[i] = len(keys) * np.log(len(loader.cat["RA"])) + 2 * fmin[i]

    samples = {key: x[:, i][mask] for i, key in enumerate(keys)}

    mean = [np.mean(samples[key]) for key in keys]
    std = [(len(samples[key] - 1) * np.var(samples[key], ddof=0))**0.5
           for key in keys]
    stats = {key: (mean[i], std[i]) for i, key in enumerate(keys)}

    loader.reset_mask()
    return samples, stats, fmin, logz, bic
